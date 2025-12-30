// ===================================================
// In file: cgadimpl/include/ad/cuda_graphs.hpp
// ===================================================
#pragma once
#include <cuda_runtime.h>

extern "C" { typedef struct CUstream_st* ag_cuda_stream_t; }

namespace ag {


inline ag_cuda_stream_t& _get_stream() {
    static thread_local ag_cuda_stream_t s = nullptr;
    return s;
}

inline ag_cuda_stream_t current_stream() {
    return _get_stream();
}

inline void set_current_stream(ag_cuda_stream_t s) {
    _get_stream() = s;
}


class CudaGraphRunner {
public:
    CudaGraphRunner();
    ~CudaGraphRunner();

    // Disallow copying to prevent issues with CUDA resources
    CudaGraphRunner(const CudaGraphRunner&) = delete;
    CudaGraphRunner& operator=(const CudaGraphRunner&) = delete;

    /**
     * @brief Puts the framework into capture mode on a private stream.
     * All subsequent CUDA operations will be recorded into the graph.
     */
    void begin_capture();

    /**
     * @brief Stops capturing and instantiates the graph for execution.
     * Resets the framework's current stream back to the default.
     */
    void end_capture();

    /**constexpr __host__ int my_function() { ... }
     * @brief Launches the entire captured graph with a single call.
     * This is the high-performance replay function.
     * @return True if replay was successful, false otherwise.
     */
    bool replay();

    cudaStream_t get_stream() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t instance_ = nullptr;
    bool is_capturing_ = false;
};

} // namespace ag