// ============================================
// cgadimpl/src/kernel_stuff/runtime.cpp
// ============================================
#include "ad/runtime/runtime.hpp"
// After (Correct, Thread-Local Implementation)
namespace ag {
  // Use 'thread_local' to make the variable specific to each thread
  static thread_local ag_cuda_stream_t g_stream = nullptr; 

  ag_cuda_stream_t current_stream() { return g_stream; }
  void set_current_stream(ag_cuda_stream_t s) { g_stream = s; }
}
