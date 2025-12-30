// =====================
// cgadimpl/include/ad/runtime/runtime.hpp
// =====================
#pragma once

extern "C" { typedef struct CUstream_st* ag_cuda_stream_t; }

namespace ag {
  // Thread-local stream storage
  inline ag_cuda_stream_t& get_stream_ref() {
    static thread_local ag_cuda_stream_t g_stream = nullptr;
    return g_stream;
  }

  // Get the stream ops should use for CUDA launches.
  inline ag_cuda_stream_t current_stream() { 
    return get_stream_ref(); 
  }

  // Set the current stream (you'll use this later for CUDA Graph capture/replay).
  inline void set_current_stream(ag_cuda_stream_t s) { 
    get_stream_ref() = s; 
  }
}
