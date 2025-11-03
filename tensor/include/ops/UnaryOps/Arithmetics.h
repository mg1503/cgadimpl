// ============================================================
// In file: tensor/include/ops/UnaryOps/Arithmetics.h
// ============================================================
#pragma once
#include "core/Tensor.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace OwnTensor {
// ============================================================
// Out-of-place unary Arithmetics functions
// ============================================================
Tensor square(const Tensor& t);
Tensor sqrt(const Tensor& t);
Tensor negator(const Tensor& t); 
Tensor abs(const Tensor& t);
Tensor sign(const Tensor& t);
Tensor reciprocal(const Tensor& t);
// ============================================================
// In-place unary Arithmetics functions
// ============================================================
void square_(Tensor& t);
void sqrt_(Tensor& t);
void negator_(Tensor& t); 
void abs_(Tensor& t); 
void sign_(Tensor& t);
void reciprocal_(Tensor& t);

// Out-of-place power functions
Tensor pow(const Tensor& t, int exponent);
Tensor pow(const Tensor& t, float exponent);
Tensor pow(const Tensor& t, double exponent);

// In-place power functions
void pow_(Tensor& t, int exponent);
void pow_(Tensor& t, float exponent);
void pow_(Tensor& t, double exponent);

#ifdef WITH_CUDA
// ============================================================
// Out-of-place unary Arithmetics functions
// ============================================================
Tensor square(const Tensor& t, cudaStream_t stream);
Tensor sqrt(const Tensor& t, cudaStream_t stream);
Tensor negator(const Tensor& t, cudaStream_t stream); 
Tensor abs(const Tensor& t, cudaStream_t stream);
Tensor sign(const Tensor& t, cudaStream_t stream);
Tensor reciprocal(const Tensor& t, cudaStream_t stream);
// ============================================================
// In-place unary Arithmetics functions
// ============================================================
void square_(Tensor& t, cudaStream_t stream);
void sqrt_(Tensor& t, cudaStream_t stream);
void negator_(Tensor& t, cudaStream_t stream); 
void abs_(Tensor& t, cudaStream_t stream); 
void sign_(Tensor& t, cudaStream_t stream);
void reciprocal_(Tensor& t, cudaStream_t stream);

// Out-of-place power functions
Tensor pow(const Tensor& t, int exponent, cudaStream_t stream);
Tensor pow(const Tensor& t, float exponent, cudaStream_t stream);
Tensor pow(const Tensor& t, double exponent, cudaStream_t stream);

// In-place power functions
void pow_(Tensor& t, int exponent, cudaStream_t stream);
void pow_(Tensor& t, float exponent, cudaStream_t stream);
void pow_(Tensor& t, double exponent, cudaStream_t stream);
#endif


} // end of namespace