# Autodiff Documentation

This document provides a detailed overview of the Automatic Differentiation (Autodiff) module within the `cgadimpl` codebase. It outlines the namespaces, functions, functionality, dependencies, and an analysis of the pros and cons of the current implementation.

## Overview

The `autodiff` module provides the core mechanics for forward and reverse mode automatic differentiation. It operates on a computational graph defined by `Value` and `Node` objects, enabling gradient computation for machine learning and optimization tasks.

## Namespaces Used

*   **`ag`**: The primary namespace for the Autodiff Graph library. All autodiff functions and core data structures reside within this namespace.

## Dependencies

The `autodiff` module relies on the following internal and external components:

| File / Component | Type | Description |
| :--- | :--- | :--- |
| `<unordered_map>` | Standard Library | Used for storing Tangent maps in Forward Mode (`jvp`) and potentially visited sets. |
| `<stdexcept>` | Standard Library | Used for error handling (e.g., throwing `std::runtime_error`). |
| `ad/ops/ops.hpp` | Internal | Defines core operator logic and likely includes `Value`/`Node` definitions. |
| `ad/detail/autodiff_ops.hpp` | Internal | Provides lookups for VJP (Vector-Jacobian Product) and JVP (Jacobian-Vector Product) functions for specific operators. |
| `ad/utils/debug.hpp` | Internal | Debugging utilities, including hooks like `on_backprop_step` and `on_jvp_step`. |
| `ad/autodiff/checkpoint.hpp`| Internal | Helper functionality for gradient checkpointing (recomputing subgraphs to save memory). |
| `ad/core/ReadyQueue.hpp` | Internal | Likely used for managing execution order or task scheduling (though `autodiff.cpp` implementation primarily uses `topo_from`). |
| `ad/core/graph.hpp` | Internal | **Crucial Dependency**. Provides `topo_from` for topological sorting of the graph. |

## Functions Declared

The following functions are declared in `include/ad/autodiff/autodiff.hpp`:

```cpp
namespace ag {

    // Resets the gradients of all nodes in the graph leading to the root.
    void zero_grad(const Value& root);

    // Performs Reverse Mode Differentiation (Backpropagation).
    void backward(const Value& root, const Tensor* grad_seed=nullptr);

    // Performs Forward Mode Differentiation (Jacobian-Vector Product).
    Tensor jvp(const Value& root, const std::unordered_map<Node*, Tensor>& seed);

} // namespace ag
```

## Functionality Breakdown

| Function | Functionality | Dependencies |
| :--- | :--- | :--- |
| **`zero_grad`** | Traverses the graph upstream from the `root` node in topological order. For every node that requires a gradient (`requires_grad()`), it resets the `.grad` tensor to zeros using `OwnTensor::Tensor::zeros`. It ensures the zero tensor matches the shape, device, and data type of the node's value. | `topo_from` (`graph.hpp`), `OwnTensor::Tensor::zeros` |
| **`backward`** | Implements **Reverse Mode Autodiff**. <br> 1. **Initialization**: Topologically sorts the graph. Initializes the `root` gradient (to `grad_seed` if provided, otherwise to 1.0 or Ones). <br> 2. **Propagation**: Iterates in *reverse* topological order. Skips nodes that do not require gradients.<br> 3. **Checkpointing**: Detects checkpointed nodes with evicted data and recomputes the subgraph.<br> 4. **VJP**: For non-leaf nodes, it looks up the VJP function (`vjp_lookup`) for the node's operator and calls it to distribute the gradient (`gy`) to its parents. | `topo_from`, `vjp_lookup` (`autodiff_ops.hpp`), `recompute_subgraph` (`checkpoint.hpp`), `OwnTensor` |
| **`jvp`** | Implements **Forward Mode Autodiff**. <br> 1. **Initialization**: Topologically sorts the graph. Creates a map `T` to store tangent vectors (gradients w.r.t input). <br> 2. **Propagation**: Iterates in forward topological order. <br> 3. **Seed**: If a node is in the `seed` map, uses the provided tangent; otherwise initializes to zeros. <br> 4. **JVP**: Looks up the JVP function (`jvp_lookup`) and computes the tangent for the current node based on parents' tangents. <br> 5. **Return**: Returns the tangent corresponding to the `root` node. | `topo_from`, `jvp_lookup` (`autodiff_ops.hpp`), `OwnTensor` |

## Analysis: Pros and Cons

### Pros
*   **Dual Mode Support**: Supports both Reverse Mode (ideal for scalar outputs like loss functions) and Forward Mode (ideal for computing Jacobian-vector products or derivatives of functions with many outputs).
*   **Gradient Checkpointing**: Explicitly handles `is_checkpoint` nodes by calling `recompute_subgraph`. This allows training larger models by trading compute for memory.
*   **Robust Tensor Initialization**: The code meticulously uses `ag::options(n->value)` to ensure that gradients and zero-initializations match the device (CPU/GPU) and dtype of the original data.
*   **Flexibility**: `backward` accepts an optional `grad_seed`, allowing backpropagation to start from arbitrary gradient signals (useful for chaining graphs or higher-order derivatives).
*   **Debuggability**: Includes hooks (`ag::debug`) to trace backpropagation steps, which is invaluable for debugging vanishing/exploding gradients.

### Cons & potential Drawbacks
*   **Memory Usage (Reverse Mode)**: The standard `backward` pass retains the entire graph and its intermediate values in memory (unless checkpointing is used). For very deep graphs, this creates significant memory pressure.
*   **Memory Leaks (Potential)**: The implementation relies on `shared_ptr` (implied by `root.node`) and raw pointers (`Node*` in `topo_from`). If the graph contains reference cycles (which is rare in pure DAG computation graphs but possible in general code), memory leaks could occur.
    *   *Note*: The `topo_from` returns a `vector<Node*>`. If `Node`s are not managed strictly by `shared_ptr` in the main graph structure, dangling pointers could be a risk, though the usage here suggests `Value` manages lifetime via `shared_ptr`.
*   **Forward Mode Overhead**: `jvp` uses a `std::unordered_map<Node*, Tensor>` to store tangents. This mimics the graph structure in a separate map. For huge graphs, this hash map lookup and storage adds overhead compared to storing dual numbers inline (though inline storage is harder to implement for dynamic graphs).
*   **recursion Risk**: `topo_from` is often implemented recursively. If the computation graph is extremely deep (e.g., unrolled RNN with thousands of steps), it could trigger a stack overflow.
*   **Thread Safety**: The `backward` function modifies `node->grad` in place. This is **not thread-safe**. You cannot perform backpropagation on the same graph instance from multiple threads simultaneously.
