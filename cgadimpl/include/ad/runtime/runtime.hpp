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

