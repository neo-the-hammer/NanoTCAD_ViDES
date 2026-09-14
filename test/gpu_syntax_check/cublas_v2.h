/* Minimal stand-in for cublas_v2.h -- see cuda_runtime.h in this
 * directory for why this exists.  Not usable for a real build.
 */
#ifndef VIDES_FAKE_CUBLAS_V2_H
#define VIDES_FAKE_CUBLAS_V2_H

#include "cuda_runtime.h"

typedef enum { CUBLAS_STATUS_SUCCESS = 0, CUBLAS_STATUS_FAILURE = 1 } cublasStatus_t;
typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 } cublasOperation_t;
typedef struct cublasContext *cublasHandle_t;

cublasStatus_t cublasCreate(cublasHandle_t *h);
cublasStatus_t cublasDestroy(cublasHandle_t h);

cublasStatus_t cublasZgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const cuDoubleComplex *alpha,
    const cuDoubleComplex *A, int lda, long long int strideA,
    const cuDoubleComplex *B, int ldb, long long int strideB,
    const cuDoubleComplex *beta,
    cuDoubleComplex *C, int ldc, long long int strideC, int batchCount);

cublasStatus_t cublasZgetrfBatched(
    cublasHandle_t handle, int n, cuDoubleComplex *const A[], int lda,
    int *P, int *info, int batchSize);

cublasStatus_t cublasZgetriBatched(
    cublasHandle_t handle, int n, const cuDoubleComplex *const A[], int lda,
    const int *P, cuDoubleComplex *const C[], int ldc, int *info,
    int batchSize);

#endif
