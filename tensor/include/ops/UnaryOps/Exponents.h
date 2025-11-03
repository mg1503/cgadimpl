// ============================================================
// In file: tensor/include/ops/UnaryOps/Exponents.h
// ============================================================
#pragma once

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "core/Tensor.h"
namespace OwnTensor{
// exponentials and logarithms
// ============================================================
// Out-of-place unary trigonometric functions
// ============================================================
Tensor exp(const Tensor& input);
Tensor exp2(const Tensor& input);
Tensor log(const Tensor& input);
Tensor log2(const Tensor& input);
Tensor log10(const Tensor& input);

// ============================================================
// In-place unary trigonometric functions
// ============================================================
void exp_(Tensor& input);
void exp2_(Tensor& input);
void log_(Tensor& input);
void log2_(Tensor& input);
void log10_(Tensor& input);


#ifdef WITH_CUDA
// exponentials and logarithms
// ============================================================
// Out-of-place unary trigonometric functions
// ============================================================
Tensor exp(const Tensor& input, cudaStream_t stream);
Tensor exp2(const Tensor& input, cudaStream_t stream);
Tensor log(const Tensor& input, cudaStream_t stream);
Tensor log2(const Tensor& input, cudaStream_t stream);
Tensor log10(const Tensor& input, cudaStream_t stream);

// ============================================================
// In-place unary trigonometric functions
// ============================================================
void exp_(Tensor& input, cudaStream_t stream);
void exp2_(Tensor& input, cudaStream_t stream);
void log_(Tensor& input, cudaStream_t stream);
void log2_(Tensor& input, cudaStream_t stream);
void log10_(Tensor& input, cudaStream_t stream);
#endif
} // end of namespace