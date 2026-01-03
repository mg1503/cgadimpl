# CUDA Graphs Documentation

This document provides a detailed overview of the CUDA Graph integration within the `cgadimpl` codebase. It outlines the namespaces, classes, functions, functionality, dependencies, and an analysis of the pros and cons of the current implementation.

## Overview

The `CudaGraphRunner` class provides a high-level wrapper around NVIDIA's CUDA Graphs API. CUDA Graphs allow a sequence of kernels (operations) to be defined once and launched repeatedly as a single unit. This drastically reduces CPU-launch overhead, which is critical for smaller models or high-throughput scenarios where the CPU is the bottleneck.

## Namespaces Used

*   **`ag`**: The primary namespace for the Autodiff Graph library.

## Dependencies

The `cuda_graphs` module relies on the following internal and external components:

| File / Component | Type | Description |
| :--- | :--- | :--- |
| `<cuda_runtime.h>` | External | NVIDIA CUDA Runtime API (structures like `cudaGraph_t`, `cudaStream_t`). |
| `ad/runtime/runtime.hpp` | Internal | Provides `ag_cuda_stream_t` and `set_current_stream` to interface with the global runtime state. |
| `ad/ag_all.hpp` | Internal | Likely a master header for `ag` types. |
| `tensor.hpp` | Internal | Provides `OwnTensor` library hooks (`OwnTensor::cuda::setCurrentStream`) to ensure tensor operations record into the graph. |
| `<device/DeviceCore.h>` | External | Likely part of the tensor library or backend utilities. |

## Class: `CudaGraphRunner`

The class `ag::CudaGraphRunner` manages the lifecycle of a CUDA graph (Capture → Instantiate → Replay).

### Functions Declared

```cpp
namespace ag {

class CudaGraphRunner {
public:
    CudaGraphRunner();
    ~CudaGraphRunner();
    
    // Capture Control
    void begin_capture();
    void end_capture();
    
    // Execution
    bool replay();

private:
    cudaStream_t stream_;
    cudaGraph_t graph_;
    cudaGraphExec_t instance_;
    bool is_capturing_;
};

} // namespace ag
```

### Functionality Breakdown

| Function | Functionality | Dependencies |
| :--- | :--- | :--- |
| **`CudaGraphRunner`** (Constructor) | Creates a dedicated, non-blocking usage-specific CUDA stream (`stream_`) for recording the graph. | `cudaStreamCreateWithFlags` |
| **`~CudaGraphRunner`** (Destructor) | Cleans up CUDA resources: destroys the executable graph, source graph, and the stream. | `cudaGraphExecDestroy`, `cudaGraphDestroy` |
| **`begin_capture`** | Starts the recording process. <br> 1. Switches the global/thread-local CUDA stream to the private `stream_`. <br> 2. Calls `cudaStreamBeginCapture` in `ThreadLocal` mode. <br> This ensures all subsequent tensor operations are recorded. | `set_current_stream`, `OwnTensor::cuda::setCurrentStream`, `cudaStreamBeginCapture` |
| **`end_capture`** | Stops recording. <br> 1. Calls `cudaStreamEndCapture` to produce the `graph_` object. <br> 2. Calls `cudaGraphInstantiate` to optimize and create the executable `instance_`. <br> 3. Resets the global stream to default (`nullptr`). | `cudaStreamEndCapture`, `cudaGraphInstantiate`, `set_current_stream` |
| **`replay`** | Executes the captured graph. <br> 1. Validates the `instance_` exists. <br> 2. Launches the graph via `cudaGraphLaunch`. <br> This is significantly faster than launching individual kernels. | `cudaGraphLaunch` |

## Analysis: Pros and Cons

### Pros
*   **Performance**: Eliminates kernel launch latency (CPU overhead). For recurring patterns (like a training step), this can deliver significant speedups.
*   **Encapsulation**: Hides the complex `cudaStreamBeginCapture`/`EndCapture`/`Instantiate`/`Launch` state machine behind a simple 3-method API (`begin`, `end`, `replay`).
*   **Integration**: Correctly hooks into the rest of the system (`OwnTensor`, `runtime.hpp`) by swapping the current stream, ensuring that library calls respect the capture stream.
*   **Safety**: Uses `cudaStreamCaptureModeThreadLocal` to avoid accidentally capturing operations from other independent threads.

### Cons & Potential Drawbacks
*   **Static Graph Limitation**: CUDA Graphs are rigid. The sequence of kernels *must* remain exactly the same every time `replay()` is called.
    *   Dynamic control flow (`if/else` on CPU conditions that change per step) will break the replay.
    *   Changing input tensor shapes (dynamic shapes) will fail, as the graph is baked for specific pointer addresses and sizes (unless graph update mechanics are implemented, which are missing here).
*   **Memory Overhead**: The `cudaGraphExec_t` instance holds a copy of kernel arguments and parameters. For massive models, this consumes extra host/device memory.
*   **Debugging Difficulty**: When a graph launch fails, error messages are often opaque ("invalid argument" or "launch failed") compared to direct kernel launches where you know exactly which line failed.
*   **Destructor Panic**: In C++, throwing exceptions (or calling `exit` via `CUDA_CHECK`) inside a destructor is dangerous. If `cudaGraphDestroy` fails during stack unwinding, it could terminate the program abruptly.
