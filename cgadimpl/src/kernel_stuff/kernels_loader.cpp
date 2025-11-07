// ============================================
// cgadimpl/src/kernel_stuff/kernels_loader.cpp
// ============================================
#include "ad/kernels_api.hpp"
#include <stdexcept>
#include <string>
#include <cstdlib>   // <<< add this for std::getenv

#if defined(_WIN32)
  #include <windows.h>
  static void* ag_dlopen(const char* p){ return (void*)LoadLibraryA(p); }
  static void* ag_dlsym(void* h, const char* s){ return (void*)GetProcAddress((HMODULE)h, s); }
  static const char* ag_dlerr(){ return "LoadLibrary/GetProcAddress failed"; }
#else
  #include <dlfcn.h>
  static void* ag_dlopen(const char* p){ return dlopen(p, RTLD_NOW); }
  static void* ag_dlsym(void* h, const char* s){ return dlsym(h, s); }
  static const char* ag_dlerr(){ return dlerror(); }
#endif

namespace ag::kernels {

static Cpu g_cpu;
Cpu& cpu(){ return g_cpu; }

static Cuda g_cuda;
Cuda& cuda(){ return g_cuda; }

void load_cpu_plugin(const char* path) {
  if (!path) throw std::runtime_error("load_cpu_plugin: null path");

  void* handle = ag_dlopen(path);
  if (!handle) throw std::runtime_error(std::string("dlopen failed: ") + ag_dlerr());

  using getter_t = int(*)(ag_cpu_v1*);
  auto sym = (getter_t)ag_dlsym(handle, "ag_get_cpu_kernels_v1");
  if (!sym) throw std::runtime_error("symbol ag_get_cpu_kernels_v1 not found");

  ag_cpu_v1 table{};
  if (sym(&table) != 0 || table.abi_version != AG_KERNELS_ABI_V1) {
    throw std::runtime_error("CPU kernels ABI mismatch or plugin init failed");
  }

  g_cpu.add    = table.add;
  g_cpu.sub    = table.sub;
  g_cpu.mul    = table.mul;
  g_cpu.div    = table.div;
  
  g_cpu.relu   = table.relu;
  g_cpu.matmul = table.matmul;
  g_cpu.gelu   = table.gelu;  
  g_cpu.leakyrelu = table.leakyrelu;
  g_cpu.sigmoid   = table.sigmoid;
  g_cpu.tanh      = table.tanh;
  g_cpu.softplus   = table.softplus;
  g_cpu.exp       = table.exp;
  g_cpu.log       = table.log;
  g_cpu.sqrt      = table.sqrt;
  g_cpu.pow       = table.pow;
  g_cpu.linear    = table.linear;

  g_cpu.relu_bwd          = table.relu_bwd;
  g_cpu.leakyrelu_bwd    = table.leakyrelu_bwd;
  g_cpu.sigmoid_bwd_from_s = table.sigmoid_bwd_from_s;
  g_cpu.tanh_bwd_from_t    = table.tanh_bwd_from_t;
  g_cpu.gelu_bwd          = table.gelu_bwd;
  g_cpu.softplus_bwd      = table.softplus_bwd;
  g_cpu.exp_bwd_from_y    = table.exp_bwd_from_y;
  g_cpu.log_bwd          = table.log_bwd;
  g_cpu.sqrt_bwd_from_y   = table.sqrt_bwd_from_y;

  // g_cpu.matmul_bwd_dA = table.matmul_bwd_dA;
  // g_cpu.matmul_bwd_dB = table.matmul_bwd_dB;
  g_cpu.linear_dW     = table.linear_dW;
  g_cpu.linear_dX     = table.linear_dX;
  g_cpu.linear_db     = table.linear_db;

}

void load_cuda_plugin(const char* path) {
  if (!path) throw std::runtime_error("load_cuda_plugin: null path");
  void* handle = ag_dlopen(path);
  if (!handle) throw std::runtime_error(std::string("dlopen failed: ") + ag_dlerr());

  using getter_t = int(*)(ag_cuda_v1*);
  auto sym = (getter_t)ag_dlsym(handle, "ag_get_cuda_kernels_v1");
  if (!sym) throw std::runtime_error("symbol ag_get_cuda_kernels_v1 not found");

  ag_cuda_v1 table{};
  if (sym(&table) != 0 || table.abi_version != AG_KERNELS_ABI_V1) {
    throw std::runtime_error("CUDA kernels ABI mismatch or plugin init failed");
  }
  g_cuda.relu   = table.relu;
  g_cuda.matmul = table.matmul;
  g_cuda.add    = table.add;
  g_cuda.exp    = table.exp;
  g_cuda.zero   = table.zero;
  g_cuda.vjp_add    = table.vjp_add;
  g_cuda.vjp_matmul = table.vjp_matmul;
  g_cuda.vjp_relu   = table.vjp_relu;
}

#ifndef AG_NO_AUTOLOAD_KERNELS
static bool try_default_autoload_cpu() {
  // Try a few common names in the current working dir
  const char* cands[] = {
#if defined(_WIN32)
    "./agkernels_cpu.dll"
#elif defined(__APPLE__)
    "./agkernels_cpu.dylib"
#else
    "./libagkernels_cpu.so"
#endif
  };
  for (const char* p : cands) {
    try { load_cpu_plugin(p); return true; } catch (...) {}
  }
  return false;
}

static bool try_default_autoload_cuda() {
  const char* cands[] = {
#if defined(_WIN32)
    "./agkernels_cuda.dll"
#elif defined(__APPLE__)
    "./agkernels_cuda.dylib"
#else
    "./libagkernels_cuda.so"
#endif
  };
  for (const char* p : cands) {
    try { load_cuda_plugin(p); return true; } catch (...) {}
  }
  return false;
}


struct AutoLoader {
  AutoLoader() {
    // CPU: env var first, else default
    if (!g_cpu.matmul && !g_cpu.relu) {
      if (const char* p = std::getenv("AG_KERNELS_CPU_PATH")) {
        try { load_cpu_plugin(p); } catch (...) {}
      } else {
        (void)try_default_autoload_cpu();
      }
    }
    // CUDA: env var first, else default
    if (!g_cuda.matmul && !g_cuda.relu) {
      if (const char* p = std::getenv("AG_KERNELS_CUDA_PATH")) {
        try { load_cuda_plugin(p); } catch (...) {}
      } else {
        (void)try_default_autoload_cuda();
      }
    }
  }
} _auto_loader;
#endif

} // namespace ag::kernels
