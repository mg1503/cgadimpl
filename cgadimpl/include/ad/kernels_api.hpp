
#pragma once
#include <cstdint>

#if defined(_WIN32)
  #define AG_EXPORT __declspec(dllexport)
  #define AG_IMPORT __declspec(dllimport)
#else
  #define AG_EXPORT __attribute__((visibility("default")))
  #define AG_IMPORT
#endif

// ---------- C ABI shared with plugins ----------
extern "C" {

// Bump when struct layout changes.
static const uint32_t AG_KERNELS_ABI_V1 = 1;

// Plain C function-pointer types (no Tensor types here)
typedef void (*ag_relu_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_matmul_fn)(const float* A, const float* B, float* C,
                             int M, int K, int N);
typedef void (*ag_gemm_fn)(const float* A, const float* B, float* C,
                             int M, int K, int N);
typedef void (*ag_gelu_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_leakyrelu_fn)(const float* x, float* y, int64_t n, float alpha);
typedef void (*ag_sigmoid_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_tanh_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_softmax_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_exp_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_log_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_sqrt_fn) (const float* x, float* y, int64_t n);
typedef void (*ag_pow_fn) (const float* x, float* y, int64_t n, float exponent);
typedef void (*ag_linear_fn)(const float* X,const float* W,const float* b,float* Y,int B,int In,int Out);
// CPU function table (can be partially filled; nulls mean "not provided")
typedef void (*elem_bwd_fn)(const float*, const float*, float*, int64_t);
typedef void (*elem_bwd_alpha_fn)(const float*, const float*, float*, int64_t, float);
typedef void (*ag_linear_dW_fn)(const float* X, const float* dY, float* dW, int B, int In, int Out);
typedef void (*ag_linear_dX_fn)(const float* dY, const float* W, float* dX, int B, int In, int Out);
typedef void (*ag_linear_db_fn)(const float* dY, float* db, int B, int Out);

    void relu_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n);
    void leakyrelu_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n, float alpha);
    void sigmoid_bwd_impl_optimized_from_s(const float* s, const float* dY, float* dX, int64_t n);
    void tanh_bwd_impl_optimized_from_t(const float* t, const float* dY, float* dX, int64_t n);
    void gelu_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n);
    void softplus_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n);
    void exp_bwd_impl_optimized_from_y(const float* y, const float* dY, float* dX, int64_t n);
    void log_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n);
    void sqrt_bwd_impl_optimized_from_y(const float* y, const float* dY, float* dX, int64_t n);
    void matmul_bwd_dA_impl_optimized(const float* dC, const float* B, float* dA, int M, int K, int N);
    void matmul_bwd_dB_impl_optimized(const float* A, const float* dC, float* dB, int M, int K, int N);
    void linear_dW_impl_optimized(const float* X, const float* dY, float* dW, int B, int In, int Out);
    void linear_dX_impl_optimized(const float* dY, const float* W, float* dX, int B, int In, int Out);
    void linear_db_impl_optimized(const float* dY, float* db, int B, int Out);

struct ag_cpu_v1 {
  uint32_t abi_version;   // must be AG_KERNELS_ABI_V1
  ag_relu_fn   relu;
  ag_matmul_fn matmul;
  ag_gemm_fn fmab;
  ag_gelu_fn gelu;
  ag_leakyrelu_fn leakyrelu;
  ag_sigmoid_fn sigmoid;
  ag_tanh_fn tanh;
  ag_softmax_fn softmax;
  ag_exp_fn exp;
  ag_log_fn log;
  ag_sqrt_fn sqrt;
  ag_pow_fn pow;
  ag_linear_fn linear;
  //backwards
  elem_bwd_fn relu_bwd;
  elem_bwd_alpha_fn leakyrelu_bwd; // takes alpha
  elem_bwd_fn sigmoid_bwd_from_s;  // if forward stored s
  elem_bwd_fn tanh_bwd_from_t;
  elem_bwd_fn gelu_bwd;
  elem_bwd_fn softplus_bwd;
  elem_bwd_fn exp_bwd_from_y;
  elem_bwd_fn log_bwd;
  elem_bwd_fn sqrt_bwd_from_y;
  // matmul backward wrappers
  void (*matmul_bwd_dA)(const float*, const float*, float*, int M, int K, int N);
  void (*matmul_bwd_dB)(const float*, const float*, float*, int M, int K, int N);
  ag_linear_dW_fn linear_dW;
  ag_linear_dX_fn linear_dX;
  ag_linear_db_fn linear_db;
};

// Every CPU plugin must export this symbol.
AG_EXPORT int ag_get_cpu_kernels_v1(struct ag_cpu_v1* out);

} // extern "C"

// ---------- C++ runtime registry & loader ----------
namespace ag::kernels {

struct Cpu {
  ag_relu_fn   relu   = nullptr;
  ag_matmul_fn matmul = nullptr;
  ag_gemm_fn fmab = nullptr;
  ag_gelu_fn gelu = nullptr;
  ag_leakyrelu_fn leakyrelu = nullptr;
  ag_sigmoid_fn sigmoid = nullptr;
  ag_tanh_fn tanh = nullptr;
  ag_softmax_fn softmax = nullptr;
  ag_exp_fn exp = nullptr;
  ag_log_fn log = nullptr;
  ag_sqrt_fn sqrt = nullptr;
  ag_pow_fn pow = nullptr;
  ag_linear_fn linear = nullptr;
  elem_bwd_fn relu_bwd = nullptr;
  elem_bwd_alpha_fn leakyrelu_bwd = nullptr;
  elem_bwd_fn sigmoid_bwd_from_s = nullptr;
  elem_bwd_fn tanh_bwd_from_t = nullptr;
  elem_bwd_fn gelu_bwd = nullptr;
  elem_bwd_fn softplus_bwd = nullptr;
  elem_bwd_fn exp_bwd_from_y = nullptr;
  elem_bwd_fn log_bwd = nullptr;
  elem_bwd_fn sqrt_bwd_from_y = nullptr;

  void (*matmul_bwd_dA)(const float*, const float*, float*, int M, int K, int N);
  void (*matmul_bwd_dB)(const float*, const float*, float*, int M, int K, int N);
  ag_linear_dW_fn linear_dW = nullptr;
  ag_linear_dX_fn linear_dX = nullptr;
  ag_linear_db_fn linear_db = nullptr;
};

// Global registry accessor
Cpu& cpu();

// Load a plugin and populate the registry
void load_cpu_plugin(const char* path);

} // namespace ag::kernels
