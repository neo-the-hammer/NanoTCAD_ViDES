/* Minimal stand-in for the CUDA runtime headers.
 *
 * This exists only so that vides_gpu.cu can be type-checked by a plain
 * C++ compiler on a machine with no CUDA toolkit installed (see
 * check_syntax.sh).  It declares just enough of the API surface that
 * vides_gpu.cu touches.  It is NOT usable for building anything real --
 * the actual build uses nvcc and the real headers.
 */
#ifndef VIDES_FAKE_CUDA_RUNTIME_H
#define VIDES_FAKE_CUDA_RUNTIME_H

#include <cstddef>
#include <cmath>

#define __global__
#define __device__
#define __host__
#define __shared__
#define __restrict__

struct vides_dim3 { unsigned int x, y, z; };
extern vides_dim3 blockIdx, blockDim, threadIdx, gridDim;
inline void __syncthreads(void) {}
inline int  atomicExch(int *addr, int val) { int o = *addr; *addr = val; return o; }

struct double2 { double x, y; };
typedef double2 cuDoubleComplex;
inline cuDoubleComplex make_cuDoubleComplex(double r, double i)
{ cuDoubleComplex z; z.x = r; z.y = i; return z; }

typedef enum { cudaSuccess = 0, cudaErrorUnknown = 1 } cudaError_t;

typedef enum {
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3
} cudaMemcpyKind;

struct cudaDeviceProp {
  char name[256];
  int  major, minor;
  size_t totalGlobalMem;
};

cudaError_t cudaMalloc(void **p, size_t n);
cudaError_t cudaFree(void *p);
cudaError_t cudaMemset(void *p, int v, size_t n);
cudaError_t cudaMemcpy(void *d, const void *s, size_t n, cudaMemcpyKind k);
cudaError_t cudaMemcpy2D(void *d, size_t dpitch, const void *s, size_t spitch,
                         size_t width, size_t height, cudaMemcpyKind k);
cudaError_t cudaGetDeviceCount(int *n);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *p, int dev);
cudaError_t cudaMemGetInfo(size_t *freeb, size_t *totalb);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaGetLastError(void);
const char *cudaGetErrorString(cudaError_t e);

#endif
