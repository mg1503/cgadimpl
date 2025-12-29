# NodeOps Documentation

This document provides a detailed overview of the Node Operations (`nodeops`) subsystem within the `cgadimpl` codebase. It specifically focuses on `nodeops.hpp` and its implementation (currently duplicated across `activation.cpp` and `nodeops.cpp`).

## Overview

The `nodeops` module is the **low-level implementation layer** for graph construction. While `ops.hpp` provides the user-facing API (e.g., `ag::add(a, b)`), `nodeops` performs the actual work of:
1.  **Computation**: Executing the immediate forward pass logic (e.g., `Tensor y = a->value + b->value`).
2.  **Graph Building**: Creating a new `Node`, linking inputs (`n->inputs = {a, b}`), and setting the `Op` type.
3.  **Tape Management**: Saving intermediate tensors required for backpropagation (e.g., `q, k, v` in Attention).

## Namespaces Used

*   **`ag`**: The primary namespace.
*   **`ag::detail`**: The namespace where all `_nodeops` functions reside, hiding them from the public API.

## Dependencies

The `nodeops` module relies on the following components:

| File / Component | Type | Description |
| :--- | :--- | :--- |
| `ad/core/graph.hpp` | Internal | Access to `Node` structure and `Value`. |
| `TensorLib.h` | Internal | **Heavy Dependency**. Uses `OwnTensor` static functions (`OwnTensor::matmul`, `OwnTensor::relu`) for actual math. |
| `ad/utils/debug.hpp` | Internal | Calls `ag::debug::on_node_created(n)` to notify the debugger/tracer. |

## Functions Declared

The functions in `include/ad/ops/nodeops.hpp` generally map 1:1 with user ops, but return `std::shared_ptr<Node>` instead of `Value`.

```cpp
namespace ag::detail {
    std::shared_ptr<Node> add_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
    std::shared_ptr<Node> matmul_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
    std::shared_ptr<Node> relu_nodeops(const std::shared_ptr<Node>& x);
    // ... many others
}
```

## Functionality Breakdown

| Function | Functionality | Dependencies |
| :--- | :--- | :--- |
| **`add_nodeops`, `sub_nodeops`** | 1. Computes `a->value + b->value`. <br> 2. Creates `Node` with `Op::Add`. <br> 3. Links inputs. | `Tensor::operator+` |
| **`matmul_nodeops`** | 1. Calls `matmul(a, b)`. <br> 2. Uses stream-aware math from `TensorLib`. | `ag::matmul` |
| **`flomul_nodeops`** | **Optimized**. Uses a `static std::unordered_map` to cache scalar Nodes for constants (e.g., `0.5`, `1.0`). Prevents creating thousands of duplicate leaf nodes for constants. | `ag::mul` |
| **`alibiatt_nodeops`** | **Complex Fused Op**. <br> 1. Computes Q, K, V. <br> 2. Generates ALIBI bias mask on CPU, moves to GPU. <br> 3. Computes Attention with Softmax. <br> 4. **Tape**: Saves `q, k, v, s` into `n->tape` for the backward pass. | `OwnTensor` |
| **`rms_nodeops`** | RMS Normalization. Calculates `x * rsqrt(mean(x^2))`. Saves `rsqrt_var` and `y` to tape. | `OwnTensor` |
| **`mambassm_nodeops`** | **State Space Model (SSM)**. <br> 1. Checks `z->tape`. <br> 2. If empty (init), computes initial state W. <br> 3. If tape has state, performs recurrent update `W_new = A*B + W_prev`. <br> 4. Updates `z->tape` for the next step. | `OwnTensor` |

## Analysis: Pros and Cons

### Pros
*   **Scalar Caching**: The implementation of `flomul` and other scalar ops uses a smart caching strategy (`static map`). This significantly reduces graph size by reusing constant nodes.
*   **Tape Management**: Complex ops like `Attention` and `RMSNorm` properly identify and save only the necessary intermediate tensors (`n->tape`), enabling efficient memory usage during backprop (standard "recompute what you can, save what you must" strategy).

### Cons, Drawbacks, and Issues
*   **Duplicate Implementation Files**: **CRITICAL**. The codebase contains both `src/ops/nodeops.cpp` (~55KB) and `src/ops/activation.cpp` (~55KB). They appear to contain identical copies of the same functions (`add_nodeops`, `relu_nodeops`, etc.). This creates a massive risk of "split brain," where a developer fixes a bug in one file but the build system compiles the other.
*   **Header Inconsistencies**: Several trigonometric functions (`asin`, `acos`, `atan`, `tan`) are **commented out** in `nodeops.hpp` but *fully implemented* in `nodeops.cpp`. This means the code exists but is unreachable/unusable by the rest of the system.
*   **Empty Files**: `src/ops/arithmetic.cpp` is essentially empty (0 bytes), despite headers suggesting a split.
*   **Hardcoded Kernels**: `alibiatt_nodeops` contains raw Loop logic for bias generation on CPU. This will be extremely slow for large sequence lengths compared to a CUDA kernel.
*   **Missing Operations**: `relumask_nodeops` throws a runtime error for CUDA (`relumask_nodeops not implemented for CUDA yet`).
*   **SSM Implementation**: The `mambassm_nodeops` implementation relies on mutating the **input node's tape** (`z->tape`) to store recurrent state. This mutability makes the graph stateful and hard to debug or re-run without clearing tapes manually.
