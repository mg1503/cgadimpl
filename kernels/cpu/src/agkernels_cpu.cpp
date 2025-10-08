// kernels/cpu/src/agkernels_cpu.cpp
#include "ad/kernels_api.hpp"
#include <cstdint>

extern "C" {

// ---------------- reference implementations ----------------
static void relu_impl(const float* x, float* y, int64_t n){
  for (int64_t i=0;i<n;++i) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

static void matmul_impl(const float* A, const float* B, float* C,
                        int M, int K, int N){
  // C(MxN) = A(MxK) * B(KxN)
  for (int i=0;i<M;++i){
    for (int j=0;j<N;++j){
      float acc = 0.f;
      const float* Ai = A + i*K;
      for (int k=0;k<K;++k) acc += Ai[k] * B[k*N + j];  
      C[i*N + j] = acc;
    }
  }
}

// ---------------- required export ----------------
AG_EXPORT int ag_get_cpu_kernels_v1(struct ag_cpu_v1* out){
  if (!out) return -1;
  out->abi_version = AG_KERNELS_ABI_V1;
  out->relu   = &relu_impl;
  out->matmul = &matmul_impl;
  return 0;
}

} // extern "C"
