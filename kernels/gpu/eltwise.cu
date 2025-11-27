// kernels/gpu/eltwise.cu
#include <cuda_runtime.h>
#include <cstdint>
#include "ad/kernels_api.hpp"

__global__ void k_relu(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { float v = x[i]; y[i] = v > 0.f ? v : 0.f; }
}
__global__ void k_exp(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __expf(x[i]);
}
__global__ void k_add(const float* __restrict__ a, const float* __restrict__ b,
                      float* __restrict__ c, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

static inline dim3 blocks_for(int64_t n, int tpb=256) {
  return dim3(int((n + tpb - 1) / tpb));
}

// NOTE: no 'static' here — we want external linkage across .cu files
void relu_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_relu<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void exp_cuda(const float* x, float* y, int64_t n, ag_cuda_stream_t s) {
  k_exp<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(x, y, n);
}
void add_cuda(const float* a, const float* b, float* c, int64_t n, ag_cuda_stream_t s) {
  k_add<<<blocks_for(n), 256, 0, (cudaStream_t)s>>>(a, b, c, n);
}
