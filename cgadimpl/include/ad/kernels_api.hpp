// cgadimpl/include/ag/kernels_api.hpp
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

// CPU function table (can be partially filled; nulls mean "not provided")
struct ag_cpu_v1 {
  uint32_t abi_version;   // must be AG_KERNELS_ABI_V1
  ag_relu_fn   relu;
  ag_matmul_fn matmul;
};

// Every CPU plugin must export this symbol.
AG_EXPORT int ag_get_cpu_kernels_v1(struct ag_cpu_v1* out);

} // extern "C"

// ---------- C++ runtime registry & loader ----------
namespace ag::kernels {

struct Cpu {
  ag_relu_fn   relu   = nullptr;
  ag_matmul_fn matmul = nullptr;
};

// Global registry accessor
Cpu& cpu();

// Load a plugin and populate the registry
void load_cpu_plugin(const char* path);

} // namespace ag::kernels
