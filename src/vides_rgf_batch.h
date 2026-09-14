// ======================================================================
//  Energy-batched Recursive Green's Function driver for NanoTCAD ViDES.
//
//  The stock code solves the NEGF problem one energy at a time:
//  CNT_charge_T / CNTmode_charge_T / GNR_charge_T / H_charge_T each run a
//  scalar `while (E<=Eupper)` loop whose body calls LDOS() -> rgfblock(),
//  a sweep of Nc block inversions and multiplications on small n x n
//  complex matrices.  Those matrices are far too small to keep a GPU busy
//  one at a time, but the energy points are completely independent of one
//  another, so the whole sweep can be run for NB energies at once.  That
//  is the axis this driver batches over.
//
//  The same entry point has two implementations:
//
//    - vides_rgf_batch_cpu()  reference, built from the stock scalar
//                             routines.  Always available.
//    - vides_rgf_batch_gpu()  cuBLAS batched implementation, compiled in
//                             only when the tree is built with GPU=1.
//
//  vides_rgf_batch() dispatches to the GPU when one is usable and falls
//  back to the CPU otherwise, so callers never need to branch.
//
//  Layout note: matrices are passed in the cmatrix() block-pointer form
//  the entry points already hold (M[i][j] for one n x n block), so the CPU
//  path forwards them untouched.  cmatrix() allocates each block from a
//  single malloc, so &M[0][0] is a contiguous row-major n*n buffer and the
//  GPU path can stage blocks straight to the device with no repacking.
//  Nothing is transposed anywhere, so CPU and GPU results are directly
//  comparable element by element.
//
//  This file is released under the BSD license, as the rest of ViDES.
//  See "license.txt".
// ======================================================================
#ifndef VIDES_RGF_BATCH_H
#define VIDES_RGF_BATCH_H

#include "complex.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Which flavour of the recursion to run.  These mirror the three
   spectral-function paths that exist in the stock tree. */
typedef enum {
  /* rgfblock() + spectralfun(): returns G[i][0] (first column) and
     G[i][Nc-1] (last column).  Used by LDOS(), i.e. by CNT_charge_T and
     GNR_charge_T and by H_charge_T in its default branch. */
  VIDES_RGF_STD  = 0,
  /* rgfblock_Lake() + spectralfun(): drops the right-going sweep, returns
     G[i][i] (diagonal) and G[i][Nc-1], and recovers A1 from the identity
     A = i(G^r - G^a) - A2.  Used by LDOS_Lake(), i.e. by the nanowire /
     Zincblend branch of H_charge_T. */
  VIDES_RGF_LAKE = 1,
  /* rgfblock() + spectralfunmode(): as STD, but the n x n mode-space
     spectral function is projected back to Nreal real-space sites through
     V A V^dagger before its diagonal is taken.  Used by LDOSMODE(), i.e.
     by CNTmode_charge_T. */
  VIDES_RGF_MODE = 2
} vides_rgf_variant;

typedef struct {
  int n;         /* order of one block                                   */
  int Nc;        /* number of blocks along transport direction           */
  int NB;        /* number of energies in this batch                     */
  int variant;   /* one of vides_rgf_variant                             */
  int flagtrans; /* if 1, also fill T[]                                  */
  double eta;    /* imaginary broadening added to E                      */

  /* MODE variant only.  Nreal is the number of real-space sites per ring
     and order[] the Nm retained mode indices, exactly as handed to
     VAVdaga().  Ignored (and may be 0/NULL) for STD and LAKE. */
  int  Nreal;
  const int *order;
} vides_rgf_desc;

/* Solve the NEGF problem for NB energies at once.
 *
 *   E        [NB]                energies, in eV
 *   diag     [Nc] of n x n       block diagonal of -H, energy independent
 *   updiag   [Nc] of n x n       upper block diagonal, energy independent
 *   lowdiag  [Nc] of n x n       lower block diagonal, energy independent
 *   sigmas   [NB] of n x n       source self-energy, one per energy
 *   sigmad   [NB] of n x n       drain  self-energy, one per energy
 *
 * The matrix arguments are in the cmatrix() block-pointer form the entry
 * points already hold, so nothing has to be repacked to call this.
 *
 *   A1       [NB][Nc][nout]      out: LDOS from the source
 *   A2       [NB][Nc][nout]      out: LDOS from the drain
 *   T        [NB]                out: transmission, untouched unless
 *                                flagtrans==1
 *
 * where nout is desc->Nreal for the MODE variant and desc->n otherwise.
 *
 * Returns 0 on success and non-zero on failure.  A non-zero return from
 * the GPU path is not fatal: vides_rgf_batch() will have already retried
 * on the CPU, so a non-zero return means the CPU path failed too.
 */
int vides_rgf_batch(const vides_rgf_desc *desc,
                    const double *E,
                    vides_complex ***diag,
                    vides_complex ***updiag,
                    vides_complex ***lowdiag,
                    vides_complex ***sigmas,
                    vides_complex ***sigmad,
                    double *A1, double *A2, double *T);

/* Reference implementation.  Dispatches per energy to the stock LDOS(),
   LDOS_Lake() or LDOSMODE(), so it reproduces the scalar path exactly and
   is the oracle the GPU is checked against. */
int vides_rgf_batch_cpu(const vides_rgf_desc *desc,
                        const double *E,
                        vides_complex ***diag,
                        vides_complex ***updiag,
                        vides_complex ***lowdiag,
                        vides_complex ***sigmas,
                        vides_complex ***sigmad,
                        double *A1, double *A2, double *T);

/* CUDA implementation.  Present in every build: when the tree is built
   without GPU=1 the stub in vides_gpu_stub.c returns non-zero so the
   dispatcher falls straight through to the CPU path. */
int vides_rgf_batch_gpu(const vides_rgf_desc *desc,
                        const double *E,
                        vides_complex ***diag,
                        vides_complex ***updiag,
                        vides_complex ***lowdiag,
                        vides_complex ***sigmas,
                        vides_complex ***sigmad,
                        double *A1, double *A2, double *T);

/* Number of energies to process per call for this problem: the GPU batch
   size when a device is in use, or a small CPU chunk otherwise.  Always
   at least 1, so callers can use it unconditionally. */
int vides_negf_chunk(const vides_rgf_desc *desc);

/* ---------------------------------------------------------------- */
/* GPU availability and policy                                       */
/* ---------------------------------------------------------------- */

/* 1 if this binary was compiled with CUDA support at all. */
int vides_gpu_compiled(void);

/* 1 if a usable CUDA device is present *and* the user has not disabled it.
   Safe to call on a CPU-only build (returns 0).  The answer is cached
   after the first call. */
int vides_gpu_available(void);

/* How many energies to put in one batch for this problem size, chosen
   from the free memory on the device.  Returns 0 if the GPU cannot hold
   even a single energy, in which case the caller should stay on the CPU. */
int vides_gpu_batch_size(const vides_rgf_desc *desc);

/* Human-readable description of the active backend, for the banner the
   entry points print.  Never NULL. */
const char *vides_gpu_describe(void);

#ifdef __cplusplus
}
#endif

#endif /* VIDES_RGF_BATCH_H */
