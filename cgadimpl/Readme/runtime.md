# Runtime Documentation

This document provides a detailed overview of the Runtime management subsystem within the `cgadimpl` codebase. It outlines the namespaces, functions, functionality, dependencies, and an analysis of the pros and cons of the current implementation.

## Overview

The `runtime` module abstracts the **execution backend state**. Specifically, it manages the concept of a "Current Stream" (typically a `cudaStream_t` for NVIDIA GPUs). This allows higher-level components (like the Autodiff Graph or JIT Compiler) to launch kernels on specific streams (e.g., for parallel branches or graph capture) without hardcoding CUDA dependencies everywhere.

## Namespaces Used

*   **`ag`**: The primary namespace for the Autodiff Graph library.

## Dependencies

The `runtime` module is minimal and has few dependencies:

| File / Component | Type | Description |
| :--- | :--- | :--- |
| `thread_local` | C++ Feature | Used to allow different threads to have different active streams (essential for multi-threaded graph capture). |
| `CUstream_st` | CUDA Type | Forward declared as an opaque struct to avoid pulling in heavy CUDA headers in the main include path. |

## Types

*   **`ag_cuda_stream_t`**: An opaque typedef for `struct CUstream_st*`. This matches the signature of a `cudaStream_t` but abstracts it away, allowing the header to be included in non-CUDA sources.

## Functions Declared

The following functions are declared in `include/ad/runtime/runtime.hpp`:

```cpp
namespace ag {

    // Get the current thread's active stream (or nullptr for default)
    ag_cuda_stream_t current_stream();

    // Set the current thread's active stream
    void set_current_stream(ag_cuda_stream_t s);

} // namespace ag
```

## Functionality Breakdown

| Function | Functionality | Dependencies |
| :--- | :--- | :--- |
| **`current_stream`** | Returns the stream associated with the current thread. This is used by `Node` creation and kernel launches to determine where operations should run. | `g_stream` (thread_local) |
| **`set_current_stream`** | Sets the global `thread_local` stream variable. This is primarily used by: <br> 1. `CudaGraphRunner::begin_capture` (to switch to a capture stream). <br> 2. `CudaGraphRunner::end_capture` (to restore the default stream). | `g_stream` (thread_local) |

## Analysis: Pros and Cons

### Pros
*   **Encapsulation**: Prevents the pollution of the global namespace with `<cuda_runtime.h>`. This drastically improves compilation speed and allows the core logic to be compiled on systems without the CUDA Toolkit installed (if needed).
*   **Thread Safety**: The use of `thread_local` (static storage duration) ensures that if multiple threads are running different models or capturing different graphs simultaneously, they do not interfere with each other's stream state.
*   **Simplicity**: The API is minimal and effectively mimics the behaviour of `cudaSetDevice` / `cudaStreamCreate` context management but for the library's internal needs.

### Cons & Potential Drawbacks
*   **Limited Scope**: Currently only manages `cudaStream_t`. It does not manage `cudaEvent_t`, device indices (active device), or handles (cublasHandle_t). A more robust runtime usually carries a full `Context` object.
*   **Opaque Type Safety**: The `typedef struct CUstream_st*` hack is standard but brittle. If the underlying CUDA API changes (unlikely for such a core type), or if a user casts a non-pointer to it, it could cause runtime crashes.
*   **Default Initialization**: Defaults to `nullptr` (Stream 0). While standard for CUDA, relying on the legacy default stream (which has synchronizing behavior) can mask concurrency bugs. Using `cudaStreamNonBlocking` streams by default would be more robust for modern asynchronous code.
