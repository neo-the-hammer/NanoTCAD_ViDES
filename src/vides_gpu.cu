// ======================================================================
//  CUDA backend for the energy-batched Recursive Green's Function sweep.
//
//  Strategy
//  --------
//  The block matrices here are small -- n is the number of atoms (or
//  modes) in one ring or slice, typically 20-200.  A single n x n complex
//  inversion is nowhere near enough work to fill a GPU, and shipping one
//  across PCIe per energy point would be slower than staying on the CPU.
//  What makes the problem a good fit is that the energy points are
//  independent: this file runs the entire Nc-step recursion for NB
//  energies simultaneously, so every cuBLAS call operates on a batch of
//  NB matrices at once.  The recursion itself stays sequential in the
//  block index, exactly as on the CPU -- only the energy axis is widened.
//
//  Layout
//  ------
//  Buffers keep the host's row-major layout (see vides_rgf_batch.h), so
//  no transposes happen anywhere.  cuBLAS is column-major, and a
//  row-major matrix X read as column-major is X^T, so:
//
//    - a product C = A*B is issued as gemm(B, A), which yields
//      B^T * A^T = (A*B)^T, i.e. exactly C in row-major.  This is the
//      same trick cmatmul.c already plays with zgemm_.
//    - an inversion needs no care at all: inv(X^T) = inv(X)^T.
//
//  Throughout, [X] denotes the column-major matrix a row-major buffer X
//  represents, so [X] = X^T and [X]^H = conj(X).
//
//  Precision
//  ---------
//  Everything is FP64, matching the CPU path.  On a GPU with a low
//  double-precision rate (Colab's default T4 runs FP64 at 1/32 of FP32)
//  the gain comes from replacing thousands of latency-bound tiny solves
//  with a few throughput-bound batched ones, not from peak FLOPs.
//
//  This file is released under the BSD license, as the rest of ViDES.
//  See "license.txt".
// ======================================================================

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Included last: vides_rgf_batch.h pulls in complex.h, which macro-aliases
   'complex' to vides_complex and would otherwise collide with the CUDA and
   C++ headers above. */
#include "vides_rgf_batch.h"

#define VIDES_PI 3.141592653589793115997963468544185161590576171875

/* vides_complex and cuDoubleComplex must be the same 16 bytes for the
   host buffers to be uploaded without repacking. */
static_assert(sizeof(vides_complex) == sizeof(cuDoubleComplex),
              "vides_complex must be layout-compatible with cuDoubleComplex");

/* ------------------------------------------------------------------ */
/* Error plumbing                                                      */
/* ------------------------------------------------------------------ */

#define CUDA_TRY(call)                                                     \
  do {                                                                     \
    cudaError_t err_ = (call);                                             \
    if (err_ != cudaSuccess) {                                             \
      fprintf(stderr, "[ViDES/GPU] %s:%d %s -> %s\n", __FILE__, __LINE__,  \
              #call, cudaGetErrorString(err_));                            \
      return -1;                                                           \
    }                                                                      \
  } while (0)

#define CUBLAS_TRY(call)                                                   \
  do {                                                                     \
    cublasStatus_t st_ = (call);                                           \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "[ViDES/GPU] %s:%d %s -> cublas status %d\n",        \
              __FILE__, __LINE__, #call, (int)st_);                        \
      return -1;                                                           \
    }                                                                      \
  } while (0)

/* ------------------------------------------------------------------ */
/* Kernels                                                             */
/* ------------------------------------------------------------------ */

static __global__ void k_fill_ptrs(cuDoubleComplex **p, cuDoubleComplex *base,
                                   size_t stride, int NB)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < NB) p[b] = base + (size_t)b * stride;
}

/* out[b] = diag_i + (E_b + i*eta) I  - (sigma ? sigma[b] : 0)
 *
 * lakeguard reproduces the |E| < 1e-10 nudge that LDOS_Lake() applies to
 * keep the recursion away from a singular point at E == 0. */
static __global__ void k_build_d(cuDoubleComplex *out,
                                 const cuDoubleComplex *diag_i,
                                 const cuDoubleComplex *sigma,
                                 const double *E, double eta,
                                 int n, int NB, int lakeguard)
{
  size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t nn = (size_t)n * n;
  if (k >= nn * (size_t)NB) return;

  int    b = (int)(k / nn);
  size_t p = k % nn;

  double Eb = E[b];
  if (lakeguard && fabs(Eb) < 1e-10) Eb += 1e-4;

  cuDoubleComplex v = diag_i[p];
  if ((p / n) == (p % n)) { v.x += Eb; v.y += eta; }
  if (sigma) {
    cuDoubleComplex s = sigma[k];
    v.x -= s.x; v.y -= s.y;
  }
  out[k] = v;
}

/* out = A - B, elementwise over the whole batch. */
static __global__ void k_sub(cuDoubleComplex *out, const cuDoubleComplex *A,
                             const cuDoubleComplex *B, size_t total)
{
  size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (k < total) { out[k].x = A[k].x - B[k].x; out[k].y = A[k].y - B[k].y; }
}

/* out = I + sgn*B, per n x n block across the batch.  sgn is -1 for the
   STD diagonal step (cmatsub against the identity) and +1 for the Lake
   one (cmatsum). */
static __global__ void k_id_axpy(cuDoubleComplex *out, const cuDoubleComplex *B,
                                 double sgn, int n, int NB)
{
  size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t nn = (size_t)n * n;
  if (k >= nn * (size_t)NB) return;
  size_t p = k % nn;
  out[k].x = sgn * B[k].x + (((p / n) == (p % n)) ? 1.0 : 0.0);
  out[k].y = sgn * B[k].y;
}

/* Negate in place, preserving the sign of anything at or below 1e-30.
   rgfblock.c guards the column recursion this way; mirrored here so the
   two paths agree on signed zeros and denormals. */
static __global__ void k_neg_guard(cuDoubleComplex *X, size_t total)
{
  size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= total) return;
  double r = X[k].x, i = X[k].y;
  X[k].x = (fabs(r) > 1e-30) ? -r : r;
  X[k].y = (fabs(i) > 1e-30) ? -i : i;
}

/* gamma = i * (S - S^dagger), matching the two-step construction in
   LDOS.c (cmatsub against cmatdaga, then multiplication by i). */
static __global__ void k_gamma(cuDoubleComplex *g, const cuDoubleComplex *S,
                               int n, int NB)
{
  size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t nn = (size_t)n * n;
  if (k >= nn * (size_t)NB) return;

  int    b  = (int)(k / nn);
  size_t p  = k % nn;
  int    i  = (int)(p / n), j = (int)(p % n);
  const cuDoubleComplex *Sb = S + (size_t)b * nn;

  double dr = Sb[(size_t)i * n + j].x - Sb[(size_t)j * n + i].x;
  double di = Sb[(size_t)i * n + j].y + Sb[(size_t)j * n + i].y;
  g[k].x = -di;
  g[k].y =  dr;
}

/* SP[b][i] = Re( sum_l G[b][i][l] * C[b][l][i] ) / (2 pi)
 * i.e. the diagonal of G*C, which is spectralfun()'s last loop. */
static __global__ void k_spectral_diag(double *SP, const cuDoubleComplex *G,
                                       const cuDoubleComplex *C,
                                       int n, int NB, int stride_out)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * NB) return;
  int b = t / n, i = t % n;
  size_t nn = (size_t)n * n;
  const cuDoubleComplex *Gb = G + (size_t)b * nn;
  const cuDoubleComplex *Cb = C + (size_t)b * nn;

  double s = 0.0;
  for (int l = 0; l < n; l++) {
    cuDoubleComplex g = Gb[(size_t)i * n + l];
    cuDoubleComplex c = Cb[(size_t)l * n + i];
    s += g.x * c.x - g.y * c.y;
  }
  SP[(size_t)b * stride_out + i] = s / (2.0 * VIDES_PI);
}

/* A1 = -2*Im(diag(Gdiag))/(2 pi) - A2, the identity A = i(G^r - G^a) - A2
   that LDOS_Lake() uses instead of a second spectral function. */
static __global__ void k_lake_A1(double *A1, const double *A2,
                                 const cuDoubleComplex *Gdiag,
                                 int n, int NB, int stride_out)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n * NB) return;
  int b = t / n, j = t % n;
  size_t nn = (size_t)n * n;
  double im = Gdiag[(size_t)b * nn + (size_t)j * n + j].y;
  size_t o  = (size_t)b * stride_out + j;
  A1[o] = -2.0 * im / (2.0 * VIDES_PI) - A2[o];
}

/* T[b] = sum_i Re( (g2 * C1)[i][i] ), the trace in transmission().
   One block per energy, reduced in shared memory. */
static __global__ void k_trace_prod(double *T, const cuDoubleComplex *g2,
                                    const cuDoubleComplex *C1, int n)
{
  extern __shared__ double sh[];
  int b = blockIdx.x;
  size_t nn = (size_t)n * n;
  const cuDoubleComplex *A = g2 + (size_t)b * nn;
  const cuDoubleComplex *B = C1 + (size_t)b * nn;

  double s = 0.0;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    for (int l = 0; l < n; l++) {
      cuDoubleComplex a = A[(size_t)i * n + l];
      cuDoubleComplex c = B[(size_t)l * n + i];
      s += a.x * c.x - a.y * c.y;
    }

  sh[threadIdx.x] = s;
  __syncthreads();
  /* blockDim.x is launched as a power of two, so this halving reduction
     is exact. */
  for (unsigned int s2 = blockDim.x / 2u; s2 > 0u; s2 >>= 1) {
    if (threadIdx.x < s2) sh[threadIdx.x] += sh[threadIdx.x + s2];
    __syncthreads();
  }
  if (threadIdx.x == 0) T[b] = sh[0];
}

/* V[i][l] = exp(i*2*pi*order[l]*i/N)/sqrt(N), the mode-to-real-space
   transform built inside VAVdaga(). */
static __global__ void k_build_V(cuDoubleComplex *V, const int *order,
                                 int Nreal, int Nm)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= Nreal * Nm) return;
  int i = t / Nm, l = t % Nm;
  double ph = 2.0 * VIDES_PI * (double)order[l] / (double)Nreal * (double)i;
  double sc = 1.0 / sqrt((double)Nreal);
  V[t].x = sc * cos(ph);
  V[t].y = sc * sin(ph);
}

/* SP[i] = Re( sum_l V[i][l] * M[l][i] ) / (2 pi), the diagonal of V*M
   that VAVdaga() feeds back to spectralfunmode(). */
static __global__ void k_mode_diag(double *SP, const cuDoubleComplex *V,
                                   const cuDoubleComplex *M,
                                   int Nreal, int Nm, int NB, int stride_out)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= Nreal * NB) return;
  int b = t / Nreal, i = t % Nreal;
  const cuDoubleComplex *Mb = M + (size_t)b * Nm * Nreal;

  double s = 0.0;
  for (int l = 0; l < Nm; l++) {
    cuDoubleComplex v = V[(size_t)i * Nm + l];
    cuDoubleComplex m = Mb[(size_t)l * Nreal + i];
    s += v.x * m.x - v.y * m.y;
  }
  SP[(size_t)b * stride_out + i] = s / (2.0 * VIDES_PI);
}

/* OR any non-zero LAPACK info code into a sticky device flag, so a
   singular block can be reported without synchronising every sweep. */
static __global__ void k_accum_info(int *flag, const int *info, int NB)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < NB && info[b] != 0) atomicExch(flag, info[b]);
}

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

#define TPB 256
static inline int nblk(size_t total) { return (int)((total + TPB - 1) / TPB); }

static const cuDoubleComplex c_one  = {1.0, 0.0};
static const cuDoubleComplex c_zero = {0.0, 0.0};

/* Row-major C = A*B over a batch.  Operands are swapped because cuBLAS
   reads the buffers column-major, i.e. transposed -- see the header
   comment.  A stride of 0 marks an operand shared by every energy, which
   is how the energy-independent Hamiltonian blocks are passed. */
static cublasStatus_t gemm_rm(cublasHandle_t h,
                              cuDoubleComplex *C, long long sC,
                              const cuDoubleComplex *A, long long sA,
                              const cuDoubleComplex *B, long long sB,
                              int n, int NB)
{
  return cublasZgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                   &c_one,
                                   B, n, sB,
                                   A, n, sA,
                                   &c_zero,
                                   C, n, sC, NB);
}

/* Row-major C = A * B^dagger over a batch.  [C] = [B]^H [A]. */
static cublasStatus_t gemm_rm_rhs_dagger(cublasHandle_t h,
                                         cuDoubleComplex *C,
                                         const cuDoubleComplex *A,
                                         const cuDoubleComplex *B,
                                         int n, int NB)
{
  long long nn = (long long)n * n;
  return cublasZgemmStridedBatched(h, CUBLAS_OP_C, CUBLAS_OP_N, n, n, n,
                                   &c_one,
                                   B, n, nn,
                                   A, n, nn,
                                   &c_zero,
                                   C, n, nn, NB);
}

/* ------------------------------------------------------------------ */
/* Workspace                                                           */
/* ------------------------------------------------------------------ */

struct Ws {
  cuDoubleComplex *diag, *up, *low;          /* shared, [Nc][n][n]      */
  cuDoubleComplex *sig_s, *sig_d, *g1, *g2;  /* per energy              */
  cuDoubleComplex *gl, *gr;                  /* [NB][Nc][n][n]          */
  cuDoubleComplex *t1, *t2, *t3, *t4;
  cuDoubleComplex *Gcur, *Gprev;             /* column recursion        */
  cuDoubleComplex *Dcur, *Dprev;             /* Lake diagonal recursion */
  cuDoubleComplex *GN0, *Cg, *Cfull, *M, *V;
  cuDoubleComplex *lu;
  cuDoubleComplex **p_lu, **p_dst;
  double *E, *A1, *A2, *T;
  int *ipiv, *info, *flag, *order;
  void **owned;
  int nowned;
};

static void ws_free(Ws *w)
{
  for (int i = 0; i < w->nowned; i++) cudaFree(w->owned[i]);
  free(w->owned);
  w->owned = NULL; w->nowned = 0;
}

/* Allocate and record, so a failure part-way through still unwinds. */
static int ws_alloc(Ws *w, void **slot, size_t bytes)
{
  if (bytes == 0) { *slot = NULL; return 0; }
  if (cudaMalloc(slot, bytes) != cudaSuccess) { *slot = NULL; return -1; }
  cudaMemset(*slot, 0, bytes);
  w->owned[w->nowned++] = *slot;
  return 0;
}

#define WS_ALLOC(field, bytes)                                             \
  do {                                                                     \
    if (ws_alloc(&w, (void **)&w.field, (bytes)) != 0) {                   \
      fprintf(stderr, "[ViDES/GPU] out of device memory allocating %s\n",  \
              #field);                                                     \
      ws_free(&w);                                                         \
      return -2;                                                           \
    }                                                                      \
  } while (0)

/* ------------------------------------------------------------------ */
/* Device discovery and batch sizing                                   */
/* ------------------------------------------------------------------ */

extern "C" int vides_gpu_compiled(void) { return 1; }

static int   g_probed = 0;
static int   g_usable = 0;
static char  g_desc[256] = "GPU not probed";

static void probe_device(void)
{
  if (g_probed) return;
  g_probed = 1;

  const char *off = getenv("VIDES_GPU");
  if (off && (off[0] == '0' || off[0] == 'n' || off[0] == 'N')) {
    snprintf(g_desc, sizeof g_desc, "GPU disabled by VIDES_GPU=%s", off);
    g_usable = 0;
    return;
  }

  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) {
    snprintf(g_desc, sizeof g_desc, "no CUDA device found");
    g_usable = 0;
    return;
  }

  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    snprintf(g_desc, sizeof g_desc, "CUDA device present but unreadable");
    g_usable = 0;
    return;
  }

  /* The FP64:FP32 ratio decides whether this is worth doing at all, so
     say it out loud rather than leaving the user to guess. */
  const char *fp64 = "";
  if (prop.major == 7 && prop.minor == 5)       fp64 = ", FP64 1/32 rate";
  else if (prop.major == 8 && prop.minor == 6)  fp64 = ", FP64 1/32 rate";
  else if (prop.major == 8 && prop.minor == 9)  fp64 = ", FP64 1/64 rate";
  else if (prop.major == 6 && prop.minor == 1)  fp64 = ", FP64 1/32 rate";

  snprintf(g_desc, sizeof g_desc, "%s (sm_%d%d, %.1f GB%s)",
           prop.name, prop.major, prop.minor,
           (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), fp64);
  g_usable = 1;
}

extern "C" int vides_gpu_available(void)
{
  probe_device();
  return g_usable;
}

extern "C" const char *vides_gpu_describe(void)
{
  probe_device();
  return g_desc;
}

extern "C" int vides_gpu_batch_size(const vides_rgf_desc *desc)
{
  probe_device();
  if (!g_usable) return 0;

  size_t freeb = 0, totalb = 0;
  if (cudaMemGetInfo(&freeb, &totalb) != cudaSuccess) return 0;

  const size_t n    = (size_t)desc->n;
  const size_t Nc   = (size_t)desc->Nc;
  const size_t nout = (size_t)((desc->variant == VIDES_RGF_MODE)
                               ? desc->Nreal : desc->n);
  const size_t nn   = n * n;
  const int    lake = (desc->variant == VIDES_RGF_LAKE);

  /* Dominant term is gl (and gr, when the right-going sweep is run);
     everything else is a fixed handful of n x n scratch blocks. */
  size_t per = (Nc * (lake ? 1 : 2) + 16) * nn * sizeof(cuDoubleComplex)
             + 2 * Nc * nout * sizeof(double)
             + sizeof(double)
             + n * sizeof(int) + 4 * sizeof(void *);

  /* The three shared Hamiltonian diagonals, plus 64 MB of headroom for
     the cuBLAS handle and its own scratch. */
  size_t shared = 3 * Nc * nn * sizeof(cuDoubleComplex) + ((size_t)64 << 20);
  if (freeb <= shared) return 0;

  size_t budget = (size_t)((double)(freeb - shared) * 0.80);
  size_t nb = budget / (per ? per : 1);

  const char *env = getenv("VIDES_GPU_BATCH");
  if (env) {
    long v = strtol(env, NULL, 10);
    if (v > 0 && (size_t)v < nb) nb = (size_t)v;
  }

  if (nb > 1024) nb = 1024;          /* past this the batched calls saturate */
  return (int)nb;
}

/* ------------------------------------------------------------------ */
/* Batched inversion                                                   */
/* ------------------------------------------------------------------ */

/* dst = inv(src) for NB matrices.  getrf overwrites its input, so src is
   staged through a scratch buffer.  Layout needs no attention here:
   inv(X^T) == inv(X)^T, so a row-major buffer inverts in place. */
static int inv_rm(cublasHandle_t h, Ws *w, cuDoubleComplex *dst,
                  const cuDoubleComplex *src, int n, int NB)
{
  size_t nn = (size_t)n * n;
  int pb = (NB + TPB - 1) / TPB;

  CUDA_TRY(cudaMemcpy(w->lu, src, (size_t)NB * nn * sizeof(cuDoubleComplex),
                      cudaMemcpyDeviceToDevice));
  k_fill_ptrs<<<pb, TPB>>>(w->p_lu,  w->lu, nn, NB);
  k_fill_ptrs<<<pb, TPB>>>(w->p_dst, dst,   nn, NB);

  CUBLAS_TRY(cublasZgetrfBatched(h, n, w->p_lu, n, w->ipiv, w->info, NB));
  k_accum_info<<<pb, TPB>>>(w->flag, w->info, NB);

  CUBLAS_TRY(cublasZgetriBatched(h, n, (const cuDoubleComplex *const *)w->p_lu,
                                 n, w->ipiv, w->p_dst, n, w->info, NB));
  k_accum_info<<<pb, TPB>>>(w->flag, w->info, NB);
  return 0;
}

/* ------------------------------------------------------------------ */
/* Driver                                                              */
/* ------------------------------------------------------------------ */

extern "C" int vides_rgf_batch_gpu(const vides_rgf_desc *desc,
                                   const double *E,
                                   vides_complex ***diag,
                                   vides_complex ***updiag,
                                   vides_complex ***lowdiag,
                                   vides_complex ***sigmas,
                                   vides_complex ***sigmad,
                                   double *A1, double *A2, double *T)
{
  probe_device();
  if (!g_usable) return -1;

  const int n   = desc->n;
  const int Nc  = desc->Nc;
  const int NB  = desc->NB;
  const int var = desc->variant;
  const int lake = (var == VIDES_RGF_LAKE);
  const int mode = (var == VIDES_RGF_MODE);
  const int nout = mode ? desc->Nreal : n;

  if (n <= 0 || Nc < 2 || NB <= 0) return -1;
  if (mode && (desc->Nreal <= 0 || desc->order == NULL)) return -1;

  const size_t nn   = (size_t)n * n;
  const size_t bnn  = (size_t)NB * nn;
  const long long sB = (long long)nn;          /* per-energy stride      */
  const long long s0 = 0;                      /* shared across energies */
  const long long sG = (long long)Nc * nn;     /* stride inside gl / gr  */
  const size_t cz   = sizeof(cuDoubleComplex);

  cublasHandle_t h = NULL;
  Ws w; memset(&w, 0, sizeof w);
  w.owned = (void **)calloc(40, sizeof(void *));
  if (!w.owned) return -2;

  WS_ALLOC(diag,  (size_t)Nc * nn * cz);
  WS_ALLOC(up,    (size_t)Nc * nn * cz);
  WS_ALLOC(low,   (size_t)Nc * nn * cz);
  WS_ALLOC(sig_s, bnn * cz);
  WS_ALLOC(sig_d, bnn * cz);
  WS_ALLOC(g1,    bnn * cz);
  WS_ALLOC(g2,    bnn * cz);
  WS_ALLOC(gl,    (size_t)NB * Nc * nn * cz);
  if (!lake) WS_ALLOC(gr, (size_t)NB * Nc * nn * cz);
  WS_ALLOC(t1, bnn * cz);  WS_ALLOC(t2, bnn * cz);
  WS_ALLOC(t3, bnn * cz);  WS_ALLOC(t4, bnn * cz);
  WS_ALLOC(Gcur,  bnn * cz);  WS_ALLOC(Gprev, bnn * cz);
  if (lake) { WS_ALLOC(Dcur, bnn * cz); WS_ALLOC(Dprev, bnn * cz); }
  WS_ALLOC(GN0,   bnn * cz);
  WS_ALLOC(Cg,    bnn * cz);
  WS_ALLOC(lu,    bnn * cz);
  if (mode) {
    WS_ALLOC(Cfull, bnn * cz);
    WS_ALLOC(M,     (size_t)NB * desc->Nreal * n * cz);
    WS_ALLOC(V,     (size_t)desc->Nreal * n * cz);
    WS_ALLOC(order, (size_t)n * sizeof(int));
  }
  WS_ALLOC(p_lu,  (size_t)NB * sizeof(cuDoubleComplex *));
  WS_ALLOC(p_dst, (size_t)NB * sizeof(cuDoubleComplex *));
  WS_ALLOC(E,     (size_t)NB * sizeof(double));
  WS_ALLOC(A1,    (size_t)NB * Nc * nout * sizeof(double));
  WS_ALLOC(A2,    (size_t)NB * Nc * nout * sizeof(double));
  WS_ALLOC(T,     (size_t)NB * sizeof(double));
  WS_ALLOC(ipiv,  (size_t)NB * n * sizeof(int));
  WS_ALLOC(info,  (size_t)NB * sizeof(int));
  WS_ALLOC(flag,  sizeof(int));

#define BAIL(rc) do { if (h) cublasDestroy(h); ws_free(&w); return (rc); } while (0)
#define TRY(expr) do { if ((expr) != 0) BAIL(-3); } while (0)

/* From here on the workspace and the cuBLAS handle are live, so the
   file-scope versions of these (which just return) would leak.  Redefine
   them to unwind first.  inv_rm() above owns nothing and keeps the plain
   versions, macros being textual. */
#undef CUDA_TRY
#undef CUBLAS_TRY
#define CUDA_TRY(call)                                                     \
  do {                                                                     \
    cudaError_t err_ = (call);                                             \
    if (err_ != cudaSuccess) {                                             \
      fprintf(stderr, "[ViDES/GPU] %s:%d %s -> %s\n", __FILE__, __LINE__,  \
              #call, cudaGetErrorString(err_));                            \
      BAIL(-1);                                                            \
    }                                                                      \
  } while (0)
#define CUBLAS_TRY(call)                                                   \
  do {                                                                     \
    cublasStatus_t st_ = (call);                                           \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "[ViDES/GPU] %s:%d %s -> cublas status %d\n",        \
              __FILE__, __LINE__, #call, (int)st_);                        \
      BAIL(-1);                                                            \
    }                                                                      \
  } while (0)

  if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) BAIL(-2);

  /* ---- upload ---------------------------------------------------- */
  /* The caller's blocks are separate cmatrix() allocations, each itself
     contiguous, so they are gathered into one staging buffer and shipped
     in a single transfer per array rather than Nc small ones. */
  {
    size_t nstage = ((size_t)Nc > (size_t)NB ? (size_t)Nc : (size_t)NB) * nn;
    cuDoubleComplex *stage = (cuDoubleComplex *)malloc(nstage * cz);
    if (!stage) BAIL(-2);

#define STAGE_UPLOAD(dev, src, count)                                        \
    do {                                                                     \
      for (int q_ = 0; q_ < (count); q_++)                                   \
        memcpy(stage + (size_t)q_ * nn, &(src)[q_][0][0], nn * cz);          \
      if (cudaMemcpy((dev), stage, (size_t)(count) * nn * cz,                \
                     cudaMemcpyHostToDevice) != cudaSuccess) {               \
        free(stage); BAIL(-2);                                               \
      }                                                                      \
    } while (0)

    STAGE_UPLOAD(w.diag,  diag,    Nc);
    STAGE_UPLOAD(w.up,    updiag,  Nc);
    STAGE_UPLOAD(w.low,   lowdiag, Nc);
    STAGE_UPLOAD(w.sig_s, sigmas,  NB);
    STAGE_UPLOAD(w.sig_d, sigmad,  NB);
#undef STAGE_UPLOAD

    free(stage);
  }

  if (cudaMemcpy(w.E, E, (size_t)NB * sizeof(double),
                 cudaMemcpyHostToDevice) != cudaSuccess)
    BAIL(-2);

  if (mode) {
    if (cudaMemcpy(w.order, desc->order, (size_t)n * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess) BAIL(-2);
    k_build_V<<<nblk((size_t)desc->Nreal * n), TPB>>>(w.V, w.order, desc->Nreal, n);
  }

  /* ---- gamma1, gamma2 -------------------------------------------- */
  k_gamma<<<nblk(bnn), TPB>>>(w.g1, w.sig_s, n, NB);
  k_gamma<<<nblk(bnn), TPB>>>(w.g2, w.sig_d, n, NB);

  /* ---- left-going sweep: gl ------------------------------------- */
  /* gl[0] = inv(d_0),  gl[i] = inv(d_i - low[i] gl[i-1] up[i-1])     */
  k_build_d<<<nblk(bnn), TPB>>>(w.t1, w.diag, w.sig_s, w.E, desc->eta, n, NB, lake);
  TRY(inv_rm(h, &w, w.gl, w.t1, n, NB));

  for (int i = 1; i < Nc; i++) {
    CUBLAS_TRY(gemm_rm(h, w.t1, sB, w.gl + (size_t)(i - 1) * nn, sG,
                       w.up + (size_t)(i - 1) * nn, s0, n, NB));
    CUBLAS_TRY(gemm_rm(h, w.t2, sB, w.low + (size_t)i * nn, s0, w.t1, sB, n, NB));
    k_build_d<<<nblk(bnn), TPB>>>(w.t3, w.diag + (size_t)i * nn,
                                  (i == Nc - 1) ? w.sig_d : NULL,
                                  w.E, desc->eta, n, NB, lake);
    k_sub<<<nblk(bnn), TPB>>>(w.t4, w.t3, w.t2, bnn);
    TRY(inv_rm(h, &w, w.gl + (size_t)i * nn, w.t4, n, NB));
  }

  /* ---- right-going sweep: gr (STD and MODE only) ----------------- */
  if (!lake) {
    k_build_d<<<nblk(bnn), TPB>>>(w.t1, w.diag + (size_t)(Nc - 1) * nn,
                                  w.sig_d, w.E, desc->eta, n, NB, 0);
    TRY(inv_rm(h, &w, w.gr + (size_t)(Nc - 1) * nn, w.t1, n, NB));

    for (int i = Nc - 2; i >= 0; i--) {
      CUBLAS_TRY(gemm_rm(h, w.t1, sB, w.gr + (size_t)(i + 1) * nn, sG,
                         w.low + (size_t)(i + 1) * nn, s0, n, NB));
      CUBLAS_TRY(gemm_rm(h, w.t2, sB, w.up + (size_t)i * nn, s0, w.t1, sB, n, NB));
      k_build_d<<<nblk(bnn), TPB>>>(w.t3, w.diag + (size_t)i * nn,
                                    (i == 0) ? w.sig_s : NULL,
                                    w.E, desc->eta, n, NB, 0);
      k_sub<<<nblk(bnn), TPB>>>(w.t4, w.t3, w.t2, bnn);
      TRY(inv_rm(h, &w, w.gr + (size_t)i * nn, w.t4, n, NB));
    }
  }

  /* ---- spectral function and transmission helpers ---------------- */
  /* Both mirror spectralfun() / spectralfunmode() / transmission():
     Cg = gamma * G^dagger, then either the diagonal of G*Cg (STD, LAKE)
     or the diagonal of V (G*Cg) V^dagger (MODE). */
  {
    const int tpb_red = 128;
    const size_t shmem = (size_t)tpb_red * sizeof(double);

#define SPECTRAL(out_base, G, GAM, STRIDE_OUT)                                 \
    do {                                                                       \
      CUBLAS_TRY(gemm_rm_rhs_dagger(h, w.Cg, (GAM), (G), n, NB));              \
      if (mode) {                                                              \
        CUBLAS_TRY(gemm_rm(h, w.Cfull, sB, (G), sB, w.Cg, sB, n, NB));         \
        CUBLAS_TRY(cublasZgemmStridedBatched(                                  \
            h, CUBLAS_OP_C, CUBLAS_OP_N, desc->Nreal, n, n, &c_one,            \
            w.V, n, 0, w.Cfull, n, sB, &c_zero,                                \
            w.M, desc->Nreal, (long long)desc->Nreal * n, NB));                \
        k_mode_diag<<<nblk((size_t)desc->Nreal * NB), TPB>>>(                  \
            (out_base), w.V, w.M, desc->Nreal, n, NB, (STRIDE_OUT));           \
      } else {                                                                 \
        k_spectral_diag<<<nblk((size_t)n * NB), TPB>>>(                        \
            (out_base), (G), w.Cg, n, NB, (STRIDE_OUT));                       \
      }                                                                        \
    } while (0)

#define TRANSMISSION(Tdev, G, GAM_IN, GAM_OUT)                                 \
    do {                                                                       \
      CUBLAS_TRY(gemm_rm_rhs_dagger(h, w.Cg, (GAM_IN), (G), n, NB));           \
      CUBLAS_TRY(gemm_rm(h, w.t1, sB, (G), sB, w.Cg, sB, n, NB));              \
      k_trace_prod<<<NB, tpb_red, shmem>>>((Tdev), (GAM_OUT), w.t1, n);        \
    } while (0)

    const int so = Nc * nout;   /* stride between energies in A1 / A2 */

    cuDoubleComplex *Gcur = w.Gcur, *Gprev = w.Gprev;
    cuDoubleComplex *Dcur = w.Dcur, *Dprev = w.Dprev, *tmpp;

    if (!lake) {
      /* ---- STD / MODE ------------------------------------------- */
      /* Seed of the first column: GREEN[0][0] =
         inv(I - gl[0] up[0] gr[1] low[1]) gl[0].  The diagonal blocks in
         between are never needed, only this one and gl[Nc-1]. */
      CUBLAS_TRY(gemm_rm(h, w.t1, sB, w.gr + nn, sG, w.low + nn, s0, n, NB));
      CUBLAS_TRY(gemm_rm(h, w.t2, sB, w.up, s0, w.t1, sB, n, NB));
      CUBLAS_TRY(gemm_rm(h, w.t3, sB, w.gl, sG, w.t2, sB, n, NB));
      k_id_axpy<<<nblk(bnn), TPB>>>(w.t4, w.t3, -1.0, n, NB);
      TRY(inv_rm(h, &w, w.t1, w.t4, n, NB));
      CUBLAS_TRY(gemm_rm(h, Gcur, sB, w.t1, sB, w.gl, sG, n, NB));

      SPECTRAL(w.A1, Gcur, w.g1, so);

      /* GREEN[i][0] = -gr[i] low[i] GREEN[i-1][0] */
      for (int i = 1; i < Nc; i++) {
        CUBLAS_TRY(gemm_rm(h, w.t1, sB, w.low + (size_t)i * nn, s0, Gcur, sB, n, NB));
        CUBLAS_TRY(gemm_rm(h, Gprev, sB, w.gr + (size_t)i * nn, sG, w.t1, sB, n, NB));
        k_neg_guard<<<nblk(bnn), TPB>>>(Gprev, bnn);
        tmpp = Gcur; Gcur = Gprev; Gprev = tmpp;
        SPECTRAL(w.A1 + (size_t)i * nout, Gcur, w.g1, so);
      }
      /* Gcur is now GREEN[Nc-1][0], what transmission() wants. */
      CUDA_TRY(cudaMemcpy(w.GN0, Gcur, bnn * cz, cudaMemcpyDeviceToDevice));

      /* GREEN[Nc-1][Nc-1] = gl[Nc-1], then
         GREEN[i][Nc-1] = -gl[i] up[i] GREEN[i+1][Nc-1] */
      CUDA_TRY(cudaMemcpy2D(Gcur, nn * cz,
                            w.gl + (size_t)(Nc - 1) * nn, (size_t)Nc * nn * cz,
                            nn * cz, NB, cudaMemcpyDeviceToDevice));
      SPECTRAL(w.A2 + (size_t)(Nc - 1) * nout, Gcur, w.g2, so);

      for (int i = Nc - 2; i >= 0; i--) {
        CUBLAS_TRY(gemm_rm(h, w.t1, sB, w.up + (size_t)i * nn, s0, Gcur, sB, n, NB));
        CUBLAS_TRY(gemm_rm(h, Gprev, sB, w.gl + (size_t)i * nn, sG, w.t1, sB, n, NB));
        k_neg_guard<<<nblk(bnn), TPB>>>(Gprev, bnn);
        tmpp = Gcur; Gcur = Gprev; Gprev = tmpp;
        SPECTRAL(w.A2 + (size_t)i * nout, Gcur, w.g2, so);
      }

      if (desc->flagtrans) TRANSMISSION(w.T, w.GN0, w.g1, w.g2);

    } else {
      /* ---- LAKE ------------------------------------------------- */
      /* No right-going sweep.  The diagonal and the last column are both
         backward recursions, so they run together in one pass, and A1
         comes from A = i(G^r - G^a) - A2 rather than a second spectral
         function. */
      CUDA_TRY(cudaMemcpy2D(Gcur, nn * cz,
                            w.gl + (size_t)(Nc - 1) * nn, (size_t)Nc * nn * cz,
                            nn * cz, NB, cudaMemcpyDeviceToDevice));
      CUDA_TRY(cudaMemcpy(Dcur, Gcur, bnn * cz, cudaMemcpyDeviceToDevice));

      SPECTRAL(w.A2 + (size_t)(Nc - 1) * nout, Gcur, w.g2, so);
      k_lake_A1<<<nblk((size_t)n * NB), TPB>>>(
          w.A1 + (size_t)(Nc - 1) * nout, w.A2 + (size_t)(Nc - 1) * nout,
          Dcur, n, NB, so);

      for (int i = Nc - 2; i >= 0; i--) {
        /* GREEN[i][i] = gl[i] (I + up[i] GREEN[i+1][i+1] low[i+1] gl[i]) */
        CUBLAS_TRY(gemm_rm(h, w.t1, sB, w.low + (size_t)(i + 1) * nn, s0,
                           w.gl + (size_t)i * nn, sG, n, NB));
        CUBLAS_TRY(gemm_rm(h, w.t2, sB, Dcur, sB, w.t1, sB, n, NB));
        CUBLAS_TRY(gemm_rm(h, w.t3, sB, w.up + (size_t)i * nn, s0, w.t2, sB, n, NB));
        k_id_axpy<<<nblk(bnn), TPB>>>(w.t4, w.t3, +1.0, n, NB);
        CUBLAS_TRY(gemm_rm(h, Dprev, sB, w.gl + (size_t)i * nn, sG, w.t4, sB, n, NB));

        /* GREEN[i][Nc-1] = -gl[i] up[i] GREEN[i+1][Nc-1] */
        CUBLAS_TRY(gemm_rm(h, w.t1, sB, w.up + (size_t)i * nn, s0, Gcur, sB, n, NB));
        CUBLAS_TRY(gemm_rm(h, Gprev, sB, w.gl + (size_t)i * nn, sG, w.t1, sB, n, NB));
        k_neg_guard<<<nblk(bnn), TPB>>>(Gprev, bnn);

        tmpp = Gcur; Gcur = Gprev; Gprev = tmpp;
        tmpp = Dcur; Dcur = Dprev; Dprev = tmpp;

        SPECTRAL(w.A2 + (size_t)i * nout, Gcur, w.g2, so);
        k_lake_A1<<<nblk((size_t)n * NB), TPB>>>(
            w.A1 + (size_t)i * nout, w.A2 + (size_t)i * nout,
            Dcur, n, NB, so);
      }

      /* LDOS_Lake() passes GREEN[0][Nc-1] with the gammas swapped. */
      if (desc->flagtrans) TRANSMISSION(w.T, Gcur, w.g2, w.g1);
    }

#undef SPECTRAL
#undef TRANSMISSION
  }

  /* ---- download and report --------------------------------------- */
  CUDA_TRY(cudaDeviceSynchronize());
  if (cudaGetLastError() != cudaSuccess) BAIL(-4);

  {
    int hflag = 0;
    CUDA_TRY(cudaMemcpy(&hflag, w.flag, sizeof(int), cudaMemcpyDeviceToHost));
    if (hflag != 0) {
      /* A singular block means the result is meaningless, so refuse the
         batch and let the dispatcher redo it on the CPU. */
      fprintf(stderr,
              "[ViDES/GPU] singular block in batched LU (info=%d); "
              "deferring this batch to the CPU\n", hflag);
      BAIL(-5);
    }
  }

  if (cudaMemcpy(A1, w.A1, (size_t)NB * Nc * nout * sizeof(double),
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(A2, w.A2, (size_t)NB * Nc * nout * sizeof(double),
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    BAIL(-2);

  if (desc->flagtrans && T) {
    if (cudaMemcpy(T, w.T, (size_t)NB * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      BAIL(-2);
  }

  cublasDestroy(h);
  ws_free(&w);
  return 0;

#undef BAIL
#undef TRY
}
