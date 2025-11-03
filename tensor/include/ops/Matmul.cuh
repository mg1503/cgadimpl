// ==================================================
// in-file: tensor/include/ops/Matmul.cuh
// ==================================================
#pragma once

#include "core/Tensor.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h> // <-- CORRECT: Outside namespace
#endif

namespace OwnTensor {
    #ifdef WITH_CUDA
        void cuda_matmul(const Tensor& A, const Tensor& B, Tensor& output, cudaStream_t stream = 0);
    #endif
}   