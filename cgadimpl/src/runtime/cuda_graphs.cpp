// ===================================================
// In file: cgadimpl/src/runtime/cuda_graphs.cpp
// ===================================================
#include "ad/runtime/cuda_graphs.hpp"
#include "ad/ag_all.hpp"
#include "tensor.hpp"
#include <iostream>
#include <device/DeviceCore.h>


// Helper macro for CUDA calls
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

namespace ag {

CudaGraphRunner::CudaGraphRunner() {
    // Create a private, non-blocking stream for capture and replay
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

CudaGraphRunner::~CudaGraphRunner() {
    if (instance_) CUDA_CHECK(cudaGraphExecDestroy(instance_));
    if (graph_) CUDA_CHECK(cudaGraphDestroy(graph_));
    if (stream_) CUDA_CHECK(cudaStreamDestroy(stream_));
}

void CudaGraphRunner::begin_capture() {
    if (is_capturing_) return;
    
    set_current_stream(reinterpret_cast<ag_cuda_stream_t>(stream_));
    OwnTensor::cuda::setCurrentStream(stream_);
    
    cudaError_t err = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) {
        std::cerr << "CAPTURE FAILED: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    is_capturing_ = true;
    std::cout << "  Capture started successfully" << std::endl;
}

void CudaGraphRunner::end_capture() {
    if (!is_capturing_) return;

    // End the capture sequence
    CUDA_CHECK(cudaStreamEndCapture(stream_, &graph_));
    is_capturing_ = false;

    // Instantiate the graph into an executable object
    // (This performs optimizations and prepares it for launch)
    CUDA_CHECK(cudaGraphInstantiate(&instance_, graph_, NULL, NULL, 0));

    // Reset the framework's global stream back to the default (nullptr)
    set_current_stream(nullptr);
    OwnTensor::cuda::setCurrentStream(nullptr);
}

bool CudaGraphRunner::replay() {
    if (!instance_) {
        std::cerr << "ERROR: Cannot replay CUDA Graph. Was it captured correctly?" << std::endl;
        return false;
    }
    // Launch the entire graph of kernels with a single, low-overhead call
    CUDA_CHECK(cudaGraphLaunch(instance_, stream_));
    
    // You might want to synchronize here in a real app to measure performance
    // CUDA_CHECK(cudaStreamSynchronize(stream_)); 
    return true;
}

} // namespace ag