// In file: cgadimpl/include/ad/cuda_graphs.hpp
#pragma once

#include "ad/runtime.hpp" // For ag_cuda_stream_t
#include <cuda_runtime.h>

namespace ag {

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

    /**
     * @brief Launches the entire captured graph with a single call.
     * This is the high-performance replay function.
     * @return True if replay was successful, false otherwise.
     */
    bool replay();

private:
    cudaStream_t stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t instance_ = nullptr;
    bool is_capturing_ = false;
};

} // namespace ag