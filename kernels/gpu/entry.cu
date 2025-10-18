// ==========================================
// kernels/gpu/entry.cu
// ==========================================
#include "ad/kernels_api.hpp"
#include <cstdint>

// Forward declarations with external linkage
extern void relu_cuda(const float*, float*, int64_t, ag_cuda_stream_t);
extern void add_cuda (const float*, const float*, float*, int64_t, ag_cuda_stream_t);
extern void exp_cuda (const float*, float*, int64_t, ag_cuda_stream_t);
extern void zero_cuda(float*, int64_t, ag_cuda_stream_t);
extern void mm_cuda  (const float*, const float*, float*, int, int, int, ag_cuda_stream_t);

extern "C" AG_EXPORT int ag_get_cuda_kernels_v1(ag_cuda_v1* out) {
  if (!out) return -1;
  out->abi_version = AG_KERNELS_ABI_V1;
  out->relu   = &relu_cuda;
  out->add    = &add_cuda;
  out->exp    = &exp_cuda;
  out->zero   = &zero_cuda;
  out->matmul = &mm_cuda;
  return 0;
}
