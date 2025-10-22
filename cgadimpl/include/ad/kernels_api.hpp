// =========================================================
// FILE: cgadimpl/include/ad/kernels_api.hpp
// =========================================================
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

// Keep existing:
static const uint32_t AG_KERNELS_ABI_V1 = 1;

// Plain C function-pointer types (no Tensor types here)
typedef void (*ag_relu_fn)(const float* x, float* y, int64_t n);
typedef void (*ag_matmul_fn)(const float* A, const float* B, float* C,
                             int M, int K, int N);

// CPU function table (can be partially filled; nulls mean "not provided")
struct ag_cpu_v1 {
  uint32_t abi_version;   // must be AG_KERNELS_ABI_V1
  ag_relu_fn   relu;
  ag_matmul_fn matmul;
};


AG_EXPORT int ag_get_cpu_kernels_v1(struct ag_cpu_v1* out);

// ---- NEW: CUDA function pointer types (accept a stream) ----
// Avoid pulling in CUDA headers here: just forward-declare the opaque type.
typedef struct CUstream_st* ag_cuda_stream_t;

typedef void (*ag_relu_cuda_fn)(const float* x, float* y, int64_t n, ag_cuda_stream_t s);
typedef void (*ag_matmul_cuda_fn)(const float* A, const float* B, float* C,
                                  int M, int K, int N, ag_cuda_stream_t s);
typedef void (*ag_add_cuda_fn)(const float* A, const float* B, float* C,
                               int64_t n, ag_cuda_stream_t s);
typedef void (*ag_exp_cuda_fn)(const float* x, float* y, int64_t n,
                               ag_cuda_stream_t s);
typedef void (*ag_zero_cuda_fn)(float* x, int64_t n, ag_cuda_stream_t s);

// NEW: VJP (backward) function types for CUDA
typedef void (*ag_vjp_add_cuda_fn)(float* gA, float* gB, const float* gy,
                                   int64_t n, ag_cuda_stream_t s);
typedef void (*ag_vjp_matmul_cuda_fn)(float* gA, float* gB, const float* gy,
                                      const float* A, const float* B,
                                      int M, int K, int N, ag_cuda_stream_t s);
typedef void (*ag_vjp_relu_cuda_fn)(float* gX, const float* gy, const float* X, int64_t n, ag_cuda_stream_t s); 

// CUDA function table
struct ag_cuda_v1 {
  uint32_t abi_version;
  // Forward ops
  ag_relu_cuda_fn   relu;
  ag_matmul_cuda_fn matmul;
  ag_add_cuda_fn    add;
  ag_exp_cuda_fn    exp;
  ag_zero_cuda_fn   zero;
  // NEW: Backward ops
  ag_vjp_add_cuda_fn    vjp_add;
  ag_vjp_matmul_cuda_fn vjp_matmul;
  ag_vjp_relu_cuda_fn   vjp_relu;  
};

// Every CUDA plugin must export this symbol.
AG_EXPORT int ag_get_cuda_kernels_v1(struct ag_cuda_v1* out);

} // extern "C"

// ---------- C++ runtime registries & loaders ----------
namespace ag::kernels {

// CPU registry (yours – unchanged)
struct Cpu {
  ag_relu_fn   relu   = nullptr;
  ag_matmul_fn matmul = nullptr;
};

// Global registry accessor
Cpu& cpu();

// Load a plugin and populate the registry
void load_cpu_plugin(const char* path);

// ---- NEW: CUDA registry ----
struct Cuda {
  // Forward
  ag_relu_cuda_fn   relu   = nullptr;
  ag_matmul_cuda_fn matmul = nullptr;
  ag_add_cuda_fn    add    = nullptr;
  ag_exp_cuda_fn    exp    = nullptr;
  ag_zero_cuda_fn   zero   = nullptr;
  
  // NEW: Backward
  ag_vjp_add_cuda_fn    vjp_add = nullptr;
  ag_vjp_matmul_cuda_fn vjp_matmul = nullptr;
  ag_vjp_relu_cuda_fn   vjp_relu   = nullptr;
};
Cuda& cuda();
void load_cuda_plugin(const char* path);

} // namespace ag::kernels