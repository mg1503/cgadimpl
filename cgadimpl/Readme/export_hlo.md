# Export HLO Documentation

This document provides a detailed overview of the HLO Interchange subsystem within the `cgadimpl` codebase. It outlines the namespaces, functions, functionality, dependencies, and an analysis of the pros and cons of the current implementation.

## Overview

The `export_hlo` module allows the computational graph to be exported to **StableHLO** (part of MLIR), a standardized intermediate representation used by compilers like XLA and IREE. This enables the custom autodiff framework to target high-performance hardware (TPUs, GPUs) via established compiler toolchains, rather than relying solely on its own runtime.

## Namespaces Used

*   **`ag`**: The primary namespace for the Autodiff Graph library.
*   **`ag::hlo`**: The dedicated namespace for HLO export utilities.

## Dependencies

The `export_hlo` module relies on the following components:

| File / Component | Type | Description |
| :--- | :--- | :--- |
| `<iostream>`, `<fstream>`, `<sstream>` | Standard Library | Used for string formatting and writing the text-based MLIR file. |
| `<unordered_map>`, `<vector>` | Standard Library | Used for mapping `Node*` to SSA value names (e.g., `%v0`). |
| `<limits>`, `<iomanip>` | Standard Library | Used for formatting float constants and infinity. |
| `ad/core/graph.hpp` | Internal | Access to the graph structure (`Node`, `Value`) to traverse and translate operations. |

## Functions Declared

The following function is declared in `include/ad/utils/export_hlo.hpp`:

```cpp
namespace ag::hlo {

    // Main Entry Point
    void dump_stablehlo(const Value& root, const std::string& filepath);

} // namespace ag::hlo
```

## Functionality Breakdown

| Function | Functionality | Dependencies |
| :--- | :--- | :--- |
| **`dump_stablehlo`** | Traverses the graph from `root` and writes a complete StableHLO `module` to `filepath`. <br> 1. **Topo Sort**: Linearizes the graph. <br> 2. **Type Printing**: Converts internal shapes (`Tensor`) to MLIR types (e.g., `tensor<32x100xf32>`). <br> 3. **Op Translation**: Switches on `n->op` and prints the corresponding `stablehlo.*` instruction (e.g., `stablehlo.add`, `stablehlo.dot_general`). <br> 4. **Broadcasting**: Automatically inserts `stablehlo.broadcast_in_dim` instructions when shapes don't align perfectly (e.g., scalar-vector ops). | `topo_from`, `hlo_type_string`, `maybe_broadcast` |
| **`hlo_type_string`** | (Internal Helper) Formats a `Tensor`'s shape and dtype into a StableHLO type string. | `Tensor::shape` |
| **`maybe_broadcast`** | (Internal Helper) Checks if a tensor needs broadcasting to match a target N-D shape. If so, emits a `stablehlo.broadcast_in_dim` op and returns the new SSA name. Handles sophisticated rank adjustments (e.g., `[B]` -> `[B, 1]`). | `Tensor::shape` |

## Analysis: Pros and Cons

### Pros
*   **Compiler Interoperability**: By outputting standard StableHLO, the framework gains access to powerful backends (XLA) without writing its own GPU/TPU kernels.
*   **N-Dimensional Support**: The implementation handles general N-D tensors, including partial broadcasting (rank expansion) and complex reductions (row-wise via `dense<[dims]>`), making it robust for real models.
*   **Complex Op Handling**: Successfully maps high-level ops like `CEWithLogits` and `Softmax` to their primitive consituents (Log, Exp, Reduce, Broadcast), effectively performing a "lowering" pass content.

### Cons & Potential Drawbacks
*   **One-Way Export**: This is strictly an exporter. There is no functionality to *import* HLO back into the `cgad` graph or execute the result directly. It requires an external tool (like `mlir-opt` or `iree-compile`) to run.
*   **String Processing Overhead**: The generation is text-based (printing strings to stream). For massive graphs (billions of nodes), this is slow and memory-intensive compared to generating an in-memory IR (like LLVM/MLIR C++ API).
*   **Hardcoded F32**: The type generation logic often hardcodes `f32` (floats) or `xf32`, potentially ignoring `double` or `int64` data types present in the source tensors, which could lead to type mismatches.
