// cgadimpl/src/kernels_loader.cpp
#include "ad/kernels_api.hpp"
#include <stdexcept>
#include <string>

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

  // Fill registry (allow partial tables)
  g_cpu.relu   = table.relu;
  g_cpu.matmul = table.matmul;
}

} // namespace ag::kernels
