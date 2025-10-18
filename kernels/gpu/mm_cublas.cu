// =========================
// kernels/gpu/mm_cublas.cu
// =========================
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "ad/kernels_api.hpp"

// external linkage (no 'static')
void mm_cuda(const float* A, const float* B, float* C,
             int M, int K, int N, ag_cuda_stream_t s) {
  static thread_local cublasHandle_t handle = nullptr;
  if (!handle) cublasCreate(&handle);
  cublasSetStream(handle, (cudaStream_t)s);

  const float alpha = 1.f, beta = 0.f;
  // Row-major trick: C(M,N) = (B^T(N,K) * A^T(K,M))^T
  cublasSgemm(handle,
              CUBLAS_OP_T, CUBLAS_OP_T,
              N, M, K,
              &alpha,
              B, K,   // lda
              A, M,   // ldb
              &beta,
              C, N);  // ldc
}
