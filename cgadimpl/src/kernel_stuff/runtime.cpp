// ============================================
// cgadimpl/src/kernel_stuff/runtime.cpp
// ============================================
#include "ad/runtime.hpp"

namespace ag {
  static ag_cuda_stream_t g_stream = nullptr; // nullptr == default CUDA stream

  ag_cuda_stream_t current_stream() { return g_stream; }
  void set_current_stream(ag_cuda_stream_t s) { g_stream = s; }
}
