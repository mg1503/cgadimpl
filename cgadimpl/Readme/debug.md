# Debug Documentation

This document provides a detailed overview of the Debugging and Visualization subsystem within the `cgadimpl` codebase. It outlines the namespaces, functions, functionality, dependencies, and an analysis of the pros and cons of the current implementation.

## Overview

The `debug` module provides a comprehensive suite of tools for inspecting the computational graph. It goes beyond simple printing, offering:
1.  **Graph Visualization**: Generating GraphViz (`.dot`) files to visualize the forward, backward (VJP), and forward-mode (JVP) graphs.
2.  **Runtime Tracing**: Hooks to print nodes as they are computed or differentiated.
3.  **Correctness Enforcers**: Surprisingly, the `on_backprop_step` hook is currently responsible for triggering **checkpoint recomputation** during the backward pass.

## Namespaces Used

*   **`ag`**: The primary namespace for the Autodiff Graph library.
*   **`ag::debug`**: The dedicated namespace for debugging tools.

## Dependencies

The `debug` module relies on the following components:

| File / Component | Type | Description |
| :--- | :--- | :--- |
| `<iostream>`, `<fstream>`, `<sstream>` | Standard Library | Used for console output and writing `.dot` files. |
| `ad/core/graph.hpp` | Internal | Access to `Node`, `Value`, and graph traversal utilities (`topo_from`). |
| `ad/autodiff/checkpoint.hpp` | Internal | **Critical Dependency**. Used in `on_backprop_step` to call `recompute_subgraph` if an input is missing. |

## Functions Declared

The following functions are declared in `include/ad/utils/debug.hpp`:

```cpp
namespace ag::debug {

    // Runtime Controls
    void enable_tracing(bool on = true);
    void set_print_limits(int max_rows, int max_cols, int width, int precision);

    // Printing
    void print_tensor(const std::string& label, const Tensor& T);
    void print_grad(const std::string& label, const Value& v);
    void print_value(const std::string& label, const Value& v);

    // Graph Inspectors
    void print_all_values(const Value& root);
    void print_all_grads(const Value& root);
    void dump_dot(const Value& root, const std::string& filepath);

    // Hooks
    void on_node_created(const std::shared_ptr<Node>& n);
    void on_backprop_step(Node* n, const Tensor& gy);
    void on_jvp_step(Node* n);

    // Specialized Visualizers
    void dump_vjp_dot(const Value& root, const std::string& filepath);
    void dump_jvp_dot(const Value& root, const std::string& filepath);
}
```

## Functionality Breakdown

| Function | Functionality | Dependencies |
| :--- | :--- | :--- |
| **`dump_dot`** | Generates a standard DOT file of the forward graph. Nodes are colored based on their gradient requirements (Leaf/Computed, ReqGrad/NoGrad). | `topo_from`, `std::ofstream` |
| **`dump_vjp_dot`** | Visualizes the **backward pass** flow. Draws red command edges from children to parents, representing the flow of `grad`. | `topo_from` |
| **`dump_jvp_dot`** | Visualizes the **forward-mode** flow. Draws green edges from parents to children, representing the flow of `tangents`. | `topo_from` |
| **`on_backprop_step`** | **CRITICAL**. Called during `autodiff::backward`. <br> 1. Tracing: Prints the node and incoming gradient. <br> 2. **Logic**: Checks if any input to the current node is a *checkpoint* that has been deleted. If so, it calls `recompute_subgraph` to restore it. | `ag::checkpoint_impl::recompute_subgraph` |
| **`on_node_created`** | Hook called by `Node` constructor. If tracing is enabled, prints the new node's generic info to stdout. | `g_trace` (global flag) |
| **`print_tensor`** | Formatted tensor printing. Uses the `Tensor::display` method (implied from complex logic) to handle device-to-host copying and formatting. | `Tensor::display` |

## Analysis: Pros and Cons

### Pros
*   **Rich Visualization**: The ability to generate three different views of the graph (Structure, VJP flow, JVP flow) is extremely valuable for educational purposes and debugging complex graph topologies.
*   **Centralized Tracing**: Instead of scattering `std::cout` inside `autodiff.cpp`, the hooks allow enabling/disabling trace logs globally.
*   **Checkpoint Safety**: It ensures that backpropagation doesn't crash when encountering deleted checkpoints by automatically restoring them.

### Cons & Potential Drawbacks
*   **Architecture Violation**: The `on_backprop_step` function is nominally a "debug" hook, but it contains **load-bearing logic** (checkpoint recomputation). If a user compiles out `debug.cpp` or disables this hook (conceptually), the autodiff engine would crash on checkpointed graphs. This logic belongs in `autodiff.cpp`, not `debug.cpp`.
*   **Global State**: Uses static global variables (`g_trace`, `g_max_r`) for configuration. This is not thread-safe if multiple threads try to debug different graphs with different settings simultaneously.
*   **IO Overhead**: The `dump_dot` functions re-traverse the entire graph (`topo_from`) and perform heavy string stream formatting. This is fine for debugging but should never be used in a production loop.
