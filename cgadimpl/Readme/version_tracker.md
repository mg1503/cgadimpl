# Version Tracker Documentation

This document provides a detailed overview of the `VersionTracker` subsystem within the `cgadimpl` codebase. It outlines the namespaces, classes, functions, functionality, dependencies, and an analysis of the pros and cons of the current implementation.

## Overview

The `VersionTracker` is a **shadow metadata system** designed to detect unsafe in-place operations in the autodiff graph. Unlike PyTorch, which embeds version counters directly into the `Tensor` object, this implementation maintains a separate (shadow) map of tensor versions. This allows adding safety checks to the autodiff engine without modifying the core linear algebra library (`OwnTensor`).

## Namespaces Used

*   **`ag`**: The primary namespace for the Autodiff Graph library.
*   **`ag::detail`**: The namespace for internal implementation details, where `VersionTracker` resides.

## Dependencies

The `VersionTracker` component is header-only (`include/ad/core/version_tracker.hpp`) and relies on the following:

| File / Component | Type | Description |
| :--- | :--- | :--- |
| `<unordered_map>` | Standard Library | Used to map tensor data pointers (`const void*`) to version integers. |
| `<mutex>` | Standard Library | Ensures thread-safe access to the global version map. |
| `tensor.hpp` | Internal | Defines the `Tensor` class whose data pointers are being tracked. |

## Class: `VersionTracker`

The class `ag::detail::VersionTracker` implements the tracking logic. It is typically accessed via the singleton accessor `ag::detail::version_tracker()`.

### Functions Declared

```cpp
namespace ag {
namespace detail {

class VersionTracker {
public:
    int get_version(const Tensor& t);
    void bump_version(const Tensor& t);
    void register_tensor(const Tensor& t);
    void unregister_tensor(const Tensor& t);
    void check_version(const Tensor& t, int expected_version, const char* op_name = "operation");
};

// Singleton Accessor
VersionTracker& version_tracker();

} // namespace detail
} // namespace ag
```

### Functionality Breakdown

| Function | Functionality | Dependencies |
| :--- | :--- | :--- |
| **`register_tensor`** | Starts tracking a tensor. Maps `t.data()` to version `0`. Should be called when a tensor is first created or introduced to the graph. | `std::unordered_map` |
| **`bump_version`** | Increments the version counter for a tensor. Must be called by any **in-place operation** (e.g., `add_`). This signals that the data has changed. | `Tensor::data()` |
| **`get_version`** | Returns the current version of a tensor. Returns `0` if the tensor is not being tracked. | `Tensor::data()` |
| **`check_version`** | Safety Critical. Compares the current version of a tensor against an `expected_version` (saved during the forward pass). Throws a `std::runtime_error` if they do not match, preventing silent gradient corruption. | `get_version`, `std::runtime_error` |
| **`unregister_tensor`** | Removes a tensor from the tracking map. Essential for preventing memory leaks in the map and preventing "address recycling" confusion (ABA problem). | `Tensor::data()` |

## Analysis: Pros and Cons

### Pros
*   **Decoupling**: It does *not* require changing the `Tensor` class layout. This is excellent for integrating with third-party tensor libraries (like `OwnTensor` seems to be) that you cannot modify.
*   **Thread Safety**: All methods are guarded by a `std::mutex`, making it safe to use in multi-threaded training environments.
*   **Simplicity**: The implementation is a straightforward hash map, easy to debug and understand.

### Cons & Potential Drawbacks
*   **Global Lock Contention**: A single global mutex guards *every* version check and update. In a high-performance, multi-threaded training loop, this will become a massive serialization bottleneck.
*   **ABA Problem / Address Reuse**: The map uses `const void*` (memory address) as the key. If a tensor is freed and a *new* tensor is allocated at the *same* address, the tracker (if not properly unregistered) might attribute the old tensor's version to the new one. Strict usage of `unregister_tensor` is required but hard to enforce manually.
*   **Memory Overhead**: The `versions_` map grows indefinitely if `unregister_tensor` is not called. There is no automatic RAII cleanup since it's a shadow map, not embedded in the `Tensor` destructor.
*   **Lookup Overhead**: Every `get_version` or `bump_version` requires a hash map lookup, which is slower than a direct member access (pointer dereference) used in embedded version counters.
