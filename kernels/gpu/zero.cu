// =========================
// kernels/gpu/zero.cu
// =========================
#include <cuda_runtime.h>
#include <cstdint>
#include "ad/kernels_api.hpp"

// external linkage (no 'static')
void zero_cuda(float* x, int64_t n, ag_cuda_stream_t s) {
  cudaMemsetAsync(x, 0, size_t(n) * sizeof(float), (cudaStream_t)s);
}
