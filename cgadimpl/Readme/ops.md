# Ops Documentation

This document provides a detailed overview of the Operations (`ops`) subsystem within the `cgadimpl` codebase. It outlines the namespaces, functions, functionality, dependencies, and an analysis of the pros and cons of the current implementation.

## Overview

The `ops` module defines the **high-level user API** for constructing the computation graph. It provides:
1.  **Operator Overloads**: Allowing users to write `z = x + y` (where `x, y` are `Value` objects).
2.  **Functional API**: Functions like `ag::matmul(a, b)`, `ag::relu(x)`.
3.  **Forward Interpreter**: The `forward_eval_node` function, which acts as the "kernel runner" for the graph, capable of executing any node given its inputs. This is crucial for checkpoint recomputation.

## Namespaces Used

*   **`ag`**: The primary namespace.
*   **`ag::detail`**: Internal namespace containing the low-level node creation logic (`_nodeops` functions), which creates `Node` objects and links them together.

## Dependencies

The `ops` module relies on the following components:

| File / Component | Type | Description |
| :--- | :--- | :--- |
| `ad/core/graph.hpp` | Internal | Defines `Value` and `Node` structures. |
| `ad/ops/nodeops.hpp` | Internal | Declares the properties of specific math operations. |
| `tensor.hpp` | Internal | Provides the actual math implementations (`OwnTensor::abs`, `Tensor::operator+`). |
| `ad/autodiff/checkpoint.hpp` | Internal | Used by the `checkpoint()` function to mark nodes. |

## Functions Declared

The functions in `include/ad/ops/ops.hpp` can be categorized as follows:

### 1. Arithmetic & Logic
*   `add`, `sub`, `mul`, `div` (and `operator+, -, *, /`)
*   `matmul`, `linear`
*   `sum`, `mean_all`, `rowsum`, `rowmax`

### 2. Activation Functions
*   `relu`, `sigmoid`, `tanh`, `softplus`, `gelu`, `silu`, `leaky_relu`
*   `exp`, `log`, `softmax_row`, `logsumexp_row`

### 3. Layers & Complex Ops
*   `attention` (Multi-head attention)
*   `alibiatt` (Attention with ALIBI bias)
*   `swiglu`, `rms` (RMSNorm), `laynor` (LayerNorm)
*   `cross_entropy_with_logits`, `mse_loss`

### 4. Graph Control
*   `checkpoint(Value, Option)`: Marks a node for recomputation.
*   `forward_eval_node(Node*)`: Recomputes a node's value.

## Functionality Breakdown

| Function | Functionality | Dependencies |
| :--- | :--- | :--- |
| **`operator+` (etc)** | Sugar for `ag::add`. Wraps the result in a `Value`. | `ag::add` |
| **`ag::add`** | Calls `ag::detail::add_nodeops` to create a new `Node` with `Op::Add` and links the inputs. Returns the new `Value`. | `add_nodeops` |
| **`checkpoint`** | Marks a generic `Value` as a checkpoint. This allows the memory manager to delete its data, knowing it can be restored via `forward_eval_node`. | `checkpoint_impl::mark_node_checkpoint` |
| **`forward_eval_node`** | **The Interpreter**. <br> 1. Takes a `Node*`. <br> 2. Switches on `node->op`. <br> 3. fetches input tensors (`node->inputs[i]->value`). <br> 4. Executes the math (e.g., `A + B` or `matmul(A, B)`). <br> 5. Returns the resulting `Tensor`. <br> **Crucial**: This function contains the *actual implementation* of complex ops like `AlibiAttention` for recomputation purposes. | `Tensor` operators |

## Analysis: Pros and Cons

### Pros
*   **Intuitive API**: The operator overloads make building graphs look exactly like writing standard math equations.
*   **Single Source of Truth (Kernels)**: `forward_eval_node` provides a centralized place where the *math* of the graph is defined for recomputation. This ensures that if a checkpoint needs to be restored, it runs exactly the same logic as the forward pass.
*   **Complex Op Support**: The implementation implies that "Fused" operations (like ALIBI Attention) are first-class citizens. `forward_eval_node` has a massive dedicated block to recompute `AlibiAttention` efficiently, showing optimization for Transformer workloads.

### Cons & Potential Drawbacks
*   **Maintenance Burden**: `forward_eval_node` is a large `switch` statement. Every time a new `Op` is added to `ops.def`, this function must be manually updated to support it, or recomputation will crash.
*   **Logic Duplication**: There appears to be a split between `operators` (which call `_nodeops`) and `forward_eval_node` (which calls `Tensor` ops). If `_nodeops` implementation differs slightly from `forward_eval_node` implementation, the "recomputed" value might differ from the "original" value, causing subtle bugs.
*   **Hardcoded Kernels**: Some ops (like `AlibiAttention` inside `ops.cpp`) contain hardcoded, detailed logic (creating bias tensors on CPU, moving to GPU, etc.). This logic is bake-in and hard to unit test in isolation.
