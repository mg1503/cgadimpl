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

  g_cpu.relu   = table.relu;
  g_cpu.matmul = table.matmul;
}

#ifndef AG_NO_AUTOLOAD_KERNELS
static bool try_default_autoload() {
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

struct AutoLoader {
  AutoLoader() {
    // <<< use g_cpu (lowercase), not G
    if (!g_cpu.matmul && !g_cpu.relu) {
      if (const char* p = std::getenv("AG_KERNELS_CPU_PATH")) {
        try { load_cpu_plugin(p); return; } catch (...) {}
      }
      (void)try_default_autoload();
    }
  }
} _auto_loader;
#endif

} // namespace ag::kernels
