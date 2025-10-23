// =============================================
// cgadimpl/include/ad/runtime.hpp
// =============================================
#pragma once

// Opaque CUDA stream type so core doesn’t include CUDA headers.
extern "C" { typedef struct CUstream_st* ag_cuda_stream_t; }

namespace ag {
  // Get the stream ops should use for CUDA launches.
  // For now (no CUDA yet) this will return nullptr = default stream.
  ag_cuda_stream_t current_stream();

  // Set the current stream (you’ll use this later for CUDA Graph capture/replay).
  void set_current_stream(ag_cuda_stream_t s);
}
// ============================================
// cgadimpl/src/kernel_stuff/runtime.cpp
// ============================================
#include "ad/runtime.hpp"

namespace ag {
  static ag_cuda_stream_t g_stream = nullptr; // nullptr == default CUDA stream

  ag_cuda_stream_t current_stream() { return g_stream; }
  void set_current_stream(ag_cuda_stream_t s) { g_stream = s; }
}
