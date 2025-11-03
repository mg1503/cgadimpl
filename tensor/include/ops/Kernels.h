// ==================================================
// in-file: tensor/include/ops/kernels.h
// ==================================================
#pragma once

#include "core/Tensor.h"
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif
namespace OwnTensor
{
    Tensor matmul(const Tensor& A, const Tensor& B);

    #ifdef WITH_CUDA
        // Asynchronous overload for high-performance use
        Tensor matmul(const Tensor& A, const Tensor& B, cudaStream_t stream = 0);
    #endif
}   



