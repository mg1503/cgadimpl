// ===================================================================
// file: tensor/src/UnaryOps/cpu/TrigonometryDispatch.cpp
// ===================================================================
#include <stdexcept>
#include "ops/UnaryOps/Trigonometry.h"
#include "core/Tensor.h"
#include <driver_types.h>

namespace OwnTensor {

// CPU backends (defined in Trigonometry.cpp)
Tensor sin_cpu  (const Tensor&); void sin__cpu  (Tensor&);
Tensor cos_cpu  (const Tensor&); void cos__cpu  (Tensor&);
Tensor tan_cpu  (const Tensor&); void tan__cpu  (Tensor&);
Tensor asin_cpu (const Tensor&); void asin__cpu (Tensor&);
Tensor acos_cpu (const Tensor&); void acos__cpu (Tensor&);
Tensor atan_cpu (const Tensor&); void atan__cpu (Tensor&);
Tensor sinh_cpu (const Tensor&); void sinh__cpu (Tensor&);
Tensor cosh_cpu (const Tensor&); void cosh__cpu (Tensor&);
Tensor tanh_cpu (const Tensor&); void tanh__cpu (Tensor&);
Tensor asinh_cpu(const Tensor&); void asinh__cpu(Tensor&);
Tensor acosh_cpu(const Tensor&); void acosh__cpu(Tensor&);
Tensor atanh_cpu(const Tensor&); void atanh__cpu(Tensor&);

// CUDA backends: real symbols in GPU builds, stubs otherwise
#ifdef WITH_CUDA
Tensor sin_cuda  (const Tensor&, cudaStream_t stream = 0); void sin__cuda  (Tensor&, cudaStream_t stream = 0);
Tensor cos_cuda  (const Tensor&, cudaStream_t stream = 0); void cos__cuda  (Tensor&, cudaStream_t stream = 0);
Tensor tan_cuda  (const Tensor&, cudaStream_t stream = 0); void tan__cuda  (Tensor&, cudaStream_t stream = 0);
Tensor asin_cuda (const Tensor&, cudaStream_t stream = 0); void asin__cuda (Tensor&, cudaStream_t stream = 0);
Tensor acos_cuda (const Tensor&, cudaStream_t stream = 0); void acos__cuda (Tensor&, cudaStream_t stream = 0);
Tensor atan_cuda (const Tensor&, cudaStream_t stream = 0); void atan__cuda (Tensor&, cudaStream_t stream = 0);
Tensor sinh_cuda (const Tensor&, cudaStream_t stream = 0); void sinh__cuda (Tensor&, cudaStream_t stream = 0);
Tensor cosh_cuda (const Tensor&, cudaStream_t stream = 0); void cosh__cuda (Tensor&, cudaStream_t stream = 0);
Tensor tanh_cuda (const Tensor&, cudaStream_t stream = 0); void tanh__cuda (Tensor&, cudaStream_t stream = 0);
Tensor asinh_cuda(const Tensor&, cudaStream_t stream = 0); void asinh__cuda(Tensor&, cudaStream_t stream = 0);
Tensor acosh_cuda(const Tensor&, cudaStream_t stream = 0); void acosh__cuda(Tensor&, cudaStream_t stream = 0);
Tensor atanh_cuda(const Tensor&, cudaStream_t stream = 0); void atanh__cuda(Tensor&, cudaStream_t stream = 0);
#else
static inline Tensor __stub_out() { throw std::runtime_error("This binary was built without CUDA support."); }
static inline void   __stub_in()  { throw std::runtime_error("This binary was built without CUDA support."); }
inline Tensor sin_cuda  (const Tensor&, cudaStream_t stream = 0) { return __stub_out(); } inline void sin__cuda  (Tensor&, cudaStream_t stream = 0) { __stub_in(); }
inline Tensor cos_cuda  (const Tensor&, cudaStream_t stream = 0) { return __stub_out(); } inline void cos__cuda  (Tensor&, cudaStream_t stream = 0) { __stub_in(); }
inline Tensor tan_cuda  (const Tensor&, cudaStream_t stream = 0) { return __stub_out(); } inline void tan__cuda  (Tensor&, cudaStream_t stream = 0) { __stub_in(); }
inline Tensor asin_cuda (const Tensor&, cudaStream_t stream = 0) { return __stub_out(); } inline void asin__cuda (Tensor&, cudaStream_t stream = 0) { __stub_in(); }
inline Tensor acos_cuda (const Tensor&, cudaStream_t stream = 0) { return __stub_out(); } inline void acos__cuda (Tensor&, cudaStream_t stream = 0) { __stub_in(); }
inline Tensor atan_cuda (const Tensor&, cudaStream_t stream = 0) { return __stub_out(); } inline void atan__cuda (Tensor&, cudaStream_t stream = 0) { __stub_in(); }
inline Tensor sinh_cuda (const Tensor&, cudaStream_t stream = 0) { return __stub_out(); } inline void sinh__cuda (Tensor&, cudaStream_t stream = 0) { __stub_in(); }
inline Tensor cosh_cuda (const Tensor&, cudaStream_t stream = 0) { return __stub_out(); } inline void cosh__cuda (Tensor&, cudaStream_t stream = 0) { __stub_in(); }
inline Tensor tanh_cuda (const Tensor&, cudaStream_t stream = 0) { return __stub_out(); } inline void tanh__cuda (Tensor&, cudaStream_t stream = 0) { __stub_in(); }
inline Tensor asinh_cuda(const Tensor&, cudaStream_t stream = 0) { return __stub_out(); } inline void asinh__cuda(Tensor&, cudaStream_t stream = 0) { __stub_in(); }
inline Tensor acosh_cuda(const Tensor&, cudaStream_t stream = 0) { return __stub_out(); } inline void acosh__cuda(Tensor&, cudaStream_t stream = 0) { __stub_in(); }
inline Tensor atanh_cuda(const Tensor&, cudaStream_t stream = 0) { return __stub_out(); } inline void atanh__cuda(Tensor&, cudaStream_t stream = 0) { __stub_in(); }
#endif

// templated helpers
// template <auto CpuFn, auto CudaFn, typename... Args>
// static inline decltype(auto) dispatch_out(const Tensor& x, Args&&... args) {
//     if (x.device().is_cuda()) return CudaFn(x, std::forward<Args>(args)...);
//     return CpuFn(x);
// }
// template <auto CpuFn, auto CudaFn, typename... Args>
// static inline void dispatch_inplace(Tensor& x, Args&&... args) {
//     if (x.device().is_cuda()) { CudaFn(x, std::forward<Args>(args)...); }
//     else                      { CpuFn (x); }
// }


template <auto CpuFn, auto CudaFn, typename... Args>
static inline decltype(auto) dispatch_out(const Tensor& x, Args&&... args) {
    if (x.device().is_cuda()) {
        // Only pass the extra arguments (the stream) to the CUDA function
        return CudaFn(x, std::forward<Args>(args)...);
    }
    // Call the CPU function with NO extra arguments
    return CpuFn(x);
}
template <auto CpuFn, auto CudaFn, typename... Args>
static inline void dispatch_inplace(Tensor& x, Args&&... args) {
    if (x.device().is_cuda()) {
        // Only pass the extra arguments (the stream) to the CUDA function
        CudaFn(x, std::forward<Args>(args)...);
    } else {
        // Call the CPU function with NO extra arguments
        CpuFn(x);
    }
}


// Out-of-place
Tensor sin   (const Tensor& x, cudaStream_t stream){ return dispatch_out<&sin_cpu,   &sin_cuda  >(x, stream); }
Tensor cos   (const Tensor& x, cudaStream_t stream){ return dispatch_out<&cos_cpu,   &cos_cuda  >(x, stream); }
Tensor tan   (const Tensor& x, cudaStream_t stream){ return dispatch_out<&tan_cpu,   &tan_cuda  >(x, stream); }
Tensor asin  (const Tensor& x, cudaStream_t stream){ return dispatch_out<&asin_cpu,  &asin_cuda >(x, stream); }
Tensor acos  (const Tensor& x, cudaStream_t stream){ return dispatch_out<&acos_cpu,  &acos_cuda >(x, stream); }
Tensor atan  (const Tensor& x, cudaStream_t stream){ return dispatch_out<&atan_cpu,  &atan_cuda >(x, stream); }
Tensor sinh  (const Tensor& x, cudaStream_t stream){ return dispatch_out<&sinh_cpu,  &sinh_cuda >(x, stream); }
Tensor cosh  (const Tensor& x, cudaStream_t stream){ return dispatch_out<&cosh_cpu,  &cosh_cuda >(x, stream); }
Tensor tanh  (const Tensor& x, cudaStream_t stream){ return dispatch_out<&tanh_cpu,  &tanh_cuda >(x, stream); }
Tensor asinh (const Tensor& x, cudaStream_t stream){ return dispatch_out<&asinh_cpu, &asinh_cuda>(x, stream); }
Tensor acosh (const Tensor& x, cudaStream_t stream){ return dispatch_out<&acosh_cpu, &acosh_cuda>(x, stream); }
Tensor atanh (const Tensor& x, cudaStream_t stream){ return dispatch_out<&atanh_cpu, &atanh_cuda>(x, stream); }

// In-place
void sin_   (Tensor& x, cudaStream_t stream){ dispatch_inplace<&sin__cpu,   &sin__cuda  >(x, stream); }
void cos_   (Tensor& x, cudaStream_t stream){ dispatch_inplace<&cos__cpu,   &cos__cuda  >(x, stream); }
void tan_   (Tensor& x, cudaStream_t stream){ dispatch_inplace<&tan__cpu,   &tan__cuda  >(x, stream); }
void asin_  (Tensor& x, cudaStream_t stream){ dispatch_inplace<&asin__cpu,  &asin__cuda >(x, stream); }
void acos_  (Tensor& x, cudaStream_t stream){ dispatch_inplace<&acos__cpu,  &acos__cuda >(x, stream); }
void atan_  (Tensor& x, cudaStream_t stream){ dispatch_inplace<&atan__cpu,  &atan__cuda >(x, stream); }
void sinh_  (Tensor& x, cudaStream_t stream){ dispatch_inplace<&sinh__cpu,  &sinh__cuda >(x, stream); }
void cosh_  (Tensor& x, cudaStream_t stream){ dispatch_inplace<&cosh__cpu,  &cosh__cuda >(x, stream); }
void tanh_  (Tensor& x, cudaStream_t stream){ dispatch_inplace<&tanh__cpu,  &tanh__cuda >(x, stream); }
void asinh_ (Tensor& x, cudaStream_t stream){ dispatch_inplace<&asinh__cpu, &asinh__cuda>(x, stream); }
void acosh_ (Tensor& x, cudaStream_t stream){ dispatch_inplace<&acosh__cpu, &acosh__cuda>(x, stream); }
void atanh_ (Tensor& x, cudaStream_t stream){ dispatch_inplace<&atanh__cpu, &atanh__cuda>(x, stream); }

} // namespace OwnTensor
