# JIT Compiler Documentation

This document provides a detailed overview of the Just-In-Time (JIT) Compiler subsystem within the `cgadimpl` codebase. It outlines the namespaces, classes, functions, functionality, dependencies, and an analysis of the pros and cons of the current implementation.

## Overview

The `jit_compiler` module provides a mechanism to **trace, optimize, and replay** computation graphs. Unlike the dynamic graph (which executes operators immediately as they are defined), the JIT compiler:
1.  **Traces** a graph of `Value`s to understand the computation flow.
2.  **Compiles** this flow into a linear `Plan` of `Step`s, resolving dependencies and memory slots.
3.  **Executes** the plan repeatedly with new inputs, avoiding the overhead of graph traversal and node allocation during inference.

## Namespaces Used

*   **`ag::jit`**: The dedicated namespace for JIT compilation structures and functions.

## Dependencies

The `jit_compiler` module relies on the following internal and external components:

| File / Component | Type | Description |
| :--- | :--- | :--- |
| `ad/core/graph.hpp` | Internal | Defines the `Value` and `Node` types used to trace the graph. |
| `ad/ops/nodeops.hpp` | Internal | Likely defines operator logic or enums (`Op`). |
| `TensorLib.h` | Internal | Access to core tensor operations (`OwnTensor::add`, `OwnTensor::grad`, etc.) and types (`Dtype`, `Device`). |
| `<variant>` | Standard Library | Used to represent different input argument types (`ArgInput`, `ArgParam`, `ArgSlot`, `ArgLit`). |

## Key Structures

### `Compiled`
A handle for a successfully compiled graph. It holds a pointer to the internal implementation (`Impl`) which contains the executable plan.

### `Plan`
The blueprint for execution.
*   **`Signature`**: Metadata (`shape`, `dtype`, `device`) for inputs and parameters to ensure they match at runtime.
*   **`steps`**: A linear sequence of instructions (`Step`) to execute.
*   **`slots`**: An abstract memory space (register file) where intermediate tensors are stored.

### `Step`
A single instruction in the plan.
*   **`op`**: The operation to perform (e.g., `Add`, `MatMul`).
*   **`args`**: List of arguments, which can be external inputs, parameters, literals, or results from previous `Step`s (slots).
*   **`out_slot`**: The index in the slot array where the result should be stored.

## Functions Declared

The following functions are declared in `include/ad/runtime/jit_compiler.hpp`:

```cpp
namespace ag::jit {

struct Compiled {
    bool run(const std::vector<Tensor*>& inputs, 
             const std::vector<Tensor*>& params, 
             Tensor& out) const;
};

Compiled compile(const Value& output,
                 const std::vector<Value>& inputs,
                 const std::vector<Value>& params,
                 const CompileOptions& opts = {});

} // namespace ag::jit
```

## Functionality Breakdown

| Function | Functionality | Dependencies |
| :--- | :--- | :--- |
| **`compile`** | The "Linker/Optimizer". <br> 1. Traces the graph from `output` backwards using `topo_from`. <br> 2. Maps input/parameter Nodes to indices. <br> 3. Flattens the DAG into a linear list of `Step`s. <br> 4. Allocates virtual "slots" for intermediate results (simple register allocation). <br> 5. Captures metadata (shape/dtype) for runtime checks. | `topo_from`, `Node`, `Plan` |
| **`Compiled::run`** | The "VM / Interpreter". <br> 1. **Validity Check**: Verifies that provided input tensors match the compiled `Signature` (shapes, devices). <br> 2. **Allocation**: Allocates a vector of `Tensor`s (slots) for intermediates. <br> 3. **Execution Loop**: Iterates through `Step`s, dispatching ops (like `OwnTensor::matmul`) using inputs from the slot array. <br> 4. **Return**: Extracts the final result from `out_slot`. | `Compiled::Impl::run`, `apply` (instruction dispatcher) |

## Analysis: Pros and Cons

### Pros
*   **Low Overhead**: Replaying a linear `std::vector<Step>` is much faster than traversing a pointer-based DAG of `std::shared_ptr<Node>`.
*   **Type Safety**: The `Signature` check ensures that the inputs provided at runtime exactly match (in shape and type) what the graph was compiled for, preventing obscure cuda errors.
*   **Independence**: The compiled `Plan` is self-contained. It doesn't rely on the original `Node` graph remaining alive, potentially allowing the original graph to be freed.
*   **Flexibility**: Supports mixed arguments: external inputs, learned parameters, intermediate results, and embedded constant literals.

### Cons & Potential Drawbacks
*   **No Dynamic Control Flow**: The `compile` phase "bakes in" a specific topological order. It cannot handle data-dependent branching (`if (tensor.sum() > 0)`).
*   **Static Shapes**: The `Signature` enforces **exact** shape matching. You cannot compile a model for batch size 32 and run it with batch size 1. You would need to recompile for every batch size.
*   **Memory Overhead (Slots)**: The `run` function allocates `std::vector<Tensor> slots`. While it moves tensors to reuse C++ objects, it doesn't implement advanced **memory planning/reuse** (like sharing the same buffer for mutually exclusive lifetimes). This might use more VRAM than necessary compared to a sophisticated allocator.
*   **Interpreter Overhead**: `Compiled::run` is essentially a bytecode interpreter. While faster than the dynamic graph, it still has C++ dispatch overhead for every op. It does not fuse kernels (unless `OwnTensor` does internally).
*   **Arg Copying**: The `ArgLit` mechanism copies literal tensors into the plan. If large constants are embedded (instead of passed as params), the `Plan` object could become huge.
