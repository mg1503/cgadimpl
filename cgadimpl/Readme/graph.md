# Graph Documentation

This document provides a detailed overview of the core Graph data structures within the `cgadimpl` codebase. It outlines the namespaces, classes, functions, functionality, dependencies, and an analysis of the pros and cons of the current implementation.

## Overview

The `graph` module defines the fundamental building blocks of the computational graph used for automatic differentiation. It introduces two primary structures:
*   **`Node`**: The actual unit of computation. It holds the data (`value`), gradients (`grad`), dependencies (`inputs`), and operation metadata.
*   **`Value`**: A lightweight wrapper around `std::shared_ptr<Node>`. This is the user-facing object that behaves like a tensor reference but automatically builds the graph.

## Namespaces Used

*   **`ag`**: The primary namespace for the Autodiff Graph library.

## Dependencies

The `graph` module relies on the following internal and external components:

| File / Component | Type | Description |
| :--- | :--- | :--- |
| `<vector>`, `<memory>` | Standard Library | Used for managing graph connections (shared pointers) and tensor shapes. |
| `<functional>`, `<unordered_set>` | Standard Library | Used in topological sort implementation. |
| `tensor.hpp` | Internal | Defines the `Tensor` class (via `OwnTensor` library likely) used for `value` and `grad`. |
| `ad/core/schema.hpp` | Internal | Defines `Op` enum (operation type) and other schema types. |
| `ad/runtime/runtime.hpp` | Internal | Provides execution context (CUDA stream, device index). |

## Key Structures

### `struct Node` (inherits `enable_shared_from_this`)
Represents a single operation or variable in the computation graph.
*   **Data**: `value` (forward pass result), `grad` (backward pass gradient).
*   **Graph**: `inputs` (parents), `is_leaf`.
*   **Metadata**: `op` (operation type), `debug_name`, `creation_context`.
*   **Memory/Checkpointing**: `saved_inputs`, `saved_rng_blob`, `is_checkpoint`.

### `struct Value`
A smart pointer wrapper that acts as a handle to a `Node`.
*   Exposes helpers like `shape()`, `val()`, `grad()`.
*   Simplifies API usage (users pass `Value`s around, not `Node*`).

## Functions Declared

The following functions are declared in `include/ad/core/graph.hpp`:

```cpp
namespace ag {

    // helper to create a leaf Value from a Tensor
    Value make_tensor(const Tensor& v, const char* name = "");

    // Topological sort
    std::vector<Node*> topo_from(Node* root);

    // Value member functions (partial list)
    // Value::Value(std::shared_ptr<Node> n);
    // Value::shape(), Value::grad(), etc.
}
```

## Functionality Breakdown

| Function | Functionality | Dependencies |
| :--- | :--- | :--- |
| **`Value::Value`** | Constructors for creating a `Value` wrapper around a `Node`. | `std::shared_ptr` |
| **`Value::val/grad`** | Accessors for the underlying `Node`'s value and gradient tensors. | `Node`, `Tensor` |
| **`Node::Node`** | Constructor. Initializes `value`. If `requires_grad` is true, it immediately allocates zero-filled memory for `grad` on the correct device. Captures the current execution stream/device. | `OwnTensor::Tensor::zeros`, `current_stream` |
| **`topo_from`** | Performs a **Depth-First Search (DFS)** to produce a topological ordering of the graph ending at `root`. Returns a list of raw `Node*` pointers where children appear after parents. used by `autodiff::backward` and `autodiff::jvp`. | `std::function`, `std::unordered_set` (visited) |
| **`make_tensor`** | Factory function to create a new "Leaf" node (parameter or input) from a raw `Tensor`. | `Node` constructor |

## Analysis: Pros and Cons

### Pros
*   **Smart Pointer Management**: The use of `std::shared_ptr<Node>` in `Value` and `Node::inputs` ensures automated memory management. Nodes are kept alive as long as children (outputs) refer to them.
*   **Lazy Gradient Allocation**: (Partially implemented) The constructor allocates gradient memory immediately if `requires_grad` is set, ensuring it is ready for accumulation.
*   **Execution Context**: The `Node` captures `creation_context` (device + stream). This is crucial for correct multi-GPU or asynchronous execution.
*   **Type Erasure Wrapper**: The `Value` struct hides the complexity of `shared_ptr` from the user API.

### Cons & Potential Drawbacks
*   **Reference Cycles**: Since `Node` uses `shared_ptr` for inputs ("parents"), the graph is a Directed Acyclic Graph (DAG). However, if a user accidentally creates a cycle (e.g., `x = x + 1` is fine as it creates a new node, but custom ops could link back), `shared_ptr` cycles would cause **memory leaks**. `weak_ptr` is not used here.
*   **Topological Sort Overhead**: `topo_from` rebuilds the global order every time it is called. It allocates a new `std::vector` and `std::unordered_set`. For massive graphs in a training loop, this allocation/deallocation per iteration is costly.
*   **Unsafe Caching**: The code contains commented-out logic for `topo_cache`. As noted in the comments, caching based on `Node*` addresses is unsafe because addresses are reused by the OS after deletion. Reactivating that cache without a unique ID system would cause crashes.
*   **Recursion Depth**: `topo_from` uses recursive DFS. A very deep computation graph (e.g. valid Recurrent Neural Net unrolled for 10k steps) will blow the C++ stack (Stack Overflow). An iterative DFS would be safer.
