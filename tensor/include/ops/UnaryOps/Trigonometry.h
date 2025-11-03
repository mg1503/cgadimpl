// ============================================================
// In file: tensor/include/ops/UnaryOps/Trigonometry.h
// ============================================================
#pragma once
#include "core/Tensor.h"
// #include <driver_types.h>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace OwnTensor
{
 
  Tensor sin  (const Tensor&);  void sin_  (Tensor&);
  Tensor cos  (const Tensor&);  void cos_  (Tensor&);
  Tensor tan  (const Tensor&);  void tan_  (Tensor&);
  Tensor asin (const Tensor&);  void asin_ (Tensor&);
  Tensor acos (const Tensor&);  void acos_ (Tensor&);
  Tensor atan (const Tensor&);  void atan_ (Tensor&);
  Tensor sinh (const Tensor&);  void sinh_ (Tensor&);
  Tensor cosh (const Tensor&);  void cosh_ (Tensor&);
  Tensor tanh (const Tensor&);  void tanh_ (Tensor&);
  Tensor asinh(const Tensor&);  void asinh_(Tensor&);
  Tensor acosh(const Tensor&);  void acosh_(Tensor&);
  Tensor atanh(const Tensor&);  void atanh_(Tensor&);

#ifdef WITH_CUDA
  Tensor sin  (const Tensor&);  void sin_  (Tensor&, cudaStream_t stream);
  Tensor cos  (const Tensor&);  void cos_  (Tensor&, cudaStream_t stream);
  Tensor tan  (const Tensor&);  void tan_  (Tensor&, cudaStream_t stream);
  Tensor asin (const Tensor&);  void asin_ (Tensor&, cudaStream_t stream);
  Tensor acos (const Tensor&);  void acos_ (Tensor&, cudaStream_t stream);
  Tensor atan (const Tensor&);  void atan_ (Tensor&, cudaStream_t stream);
  Tensor sinh (const Tensor&);  void sinh_ (Tensor&, cudaStream_t stream);
  Tensor cosh (const Tensor&);  void cosh_ (Tensor&, cudaStream_t stream);
  Tensor tanh (const Tensor&);  void tanh_ (Tensor&, cudaStream_t stream);
  Tensor asinh(const Tensor&);  void asinh_(Tensor&, cudaStream_t stream);
  Tensor acosh(const Tensor&);  void acosh_(Tensor&, cudaStream_t stream);
  Tensor atanh(const Tensor&);  void atanh_(Tensor&, cudaStream_t stream);

  #endif


}