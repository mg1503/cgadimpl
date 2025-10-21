// ====================================================================
// FILE: kernels/gpu/vjp.cu (The Final, Correct Version)
// ====================================================================
#include <cuda_runtime.h>
#include <cstdint>
#include "ad/kernels_api.hpp"

__global__ void k_vjp_add_accum(float* gA, float* gB, const float* gy, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&gA[i], gy[i]);
        atomicAdd(&gB[i], gy[i]);
    }
}

void vjp_add_cuda(float* gA, float* gB, const float* gy,
                  int64_t n, ag_cuda_stream_t s) {
    dim3 blocks( (unsigned int)((n + 255) / 256) );
    k_vjp_add_accum<<<blocks, 256, 0, (cudaStream_t)s>>>(gA, gB, gy, n);
}