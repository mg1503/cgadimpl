# Inplace Operations Documentation

This document provides a detailed overview of the Inplace Operations subsystem within the `cgadimpl` codebase. It outlines the namespaces, functions, functionality, dependencies, and an analysis of the pros and cons of the current implementation.

## Overview

The `inplace` module handles **in-place checkpointing, versioning, and alias tracking**. In-place operations (e.g., `x += y` or `x.add_(y)`) modify tensor memory directly for performance. However, this can corrupt the computation graph required for backpropagation if the original values are overwritten but later needed. This system solves that problem by:
1.  **Versioning**: Tracking how many times a tensor has been modified.
2.  **Snapshotting**: Saving copies of tensors before they are overwritten (if needed for backward pass).
3.  **Aliasing**: Tracking which tensors share the same underlying memory to propagate updates correctly.

## Namespaces Used

*   **`ag`**: The primary namespace for the Autodiff Graph library.
*   **`ag::inplace`**: The core namespace for inplace modification logic.
*   **`ag::inplace::detail`**: Helper namespace for internal memory management hooks (used by `careful_deletion`).
*   **`ag::inplace::debug`**: Namespace for debugging utilities.

## Dependencies

The `inplace` module relies on the following internal and external components:

| File / Component | Type | Description |
| :--- | :--- | :--- |
| `<unordered_map>`, `<unordered_set>` | Standard Library | Used for global tracking tables (`g_snapshots`, `g_meta`, `g_alias`). |
| `<mutex>` | Standard Library | Used to ensure thread safety for the global tracking tables. |
| `ad/core/graph.hpp` | Internal | Defines `Node`, `Value`, `Op`, and `Tensor`. |
| `ad/autodiff/checkpoint.hpp` | Internal | Used to delegate recomputation logic (`recompute_subgraph`). |
| `ad/autodiff/careful_deletion.hpp` | Internal | Referenced for interaction with memory cleanup policies. |
| `ad/utils/debug.hpp` | Internal | General debugging utilities. |

## Functions Declared

The following functions are declared in `include/ad/autodiff/inplace.hpp`:

```cpp
namespace ag {
namespace inplace {

    struct InplaceOptions { ... };

    // Core API
    void mark_inplace_checkpoint(const std::shared_ptr<Node>& node, const InplaceOptions& opts = {});
    bool ensure_inplace_value(const std::shared_ptr<Node>& node);
    bool recompute_inplace(const std::shared_ptr<Node>& node);
    void on_recomputed(Node* node);
    
    // Global Management
    void clear_inplace_checkpoints();
    size_t get_snapshot_memory_usage();
    size_t cleanup_stale_snapshots();

    // Versioning / Aliasing
    void register_tensor_alias(void* data_ptr, Node* node);
    void bump_tensor_version(Node* node);
    size_t get_tensor_version(Node* node);
    void debug_alias_table();

    namespace detail {
        bool has_alias(Node* node);
        bool erase_snapshot(Node* node);
    }
}
}
```

## Functionality Breakdown

| Function | Functionality | Dependencies |
| :--- | :--- | :--- |
| **`mark_inplace_checkpoint`** | Called before an in-place op. <br> 1. Saves a **snapshot** (copy) of the current tensor value. <br> 2. Records the current `version`. <br> 3. Saves input dependencies (like standard checkpointing). <br> purpose: enables restoration if the tensor is later overwritten. | `Node::value.clone()`, `g_snapshots` |
| **`ensure_inplace_value`** | Called when a tensor value is needed (e.g., during backward). <br> 1. Checks if value exists. <br> 2. If missing/stale, checks for a snapshot. <br> 3. If snapshot matches version, **restores** it. <br> 4. If snapshot is stale, triggers **recomputation**. | `g_snapshots`, `recompute_inplace` |
| **`recompute_inplace`** | Manually triggers recomputation for a node. <br> 1. Prevents recursive loops. <br> 2. Calls `checkpoint_impl::recompute_subgraph`. <br> 3. Updates the version and snapshot after success. <br> 4. Propagates the new value to all **aliased** nodes. | `checkpoint_impl::recompute_subgraph`, `propagate_to_aliases` |
| **`register_tensor_alias`** | Maps a raw data pointer to the Node(s) that use it. Used to track which tensors share memory (views). | `g_alias`, `g_meta` |
| **`propagate_to_aliases`** | (Internal) When a node is recomputed, this function updates the `value` of all other nodes sharing the same memory pointer, ensuring consistency across views. | `g_alias` |
| **`cleanup_stale_snapshots`** | Garbage collection for snapshots. Deletes snapshots for nodes that already have a valid, up-to-date value in memory, freeing RAM. | `g_snapshots` |

## Analysis: Pros and Cons

### Pros
*   **Correctness**: Mimics PyTorch's robust "version counter" mechanism, ensuring that in-place operations don't silently produce wrong gradients.
*   **Memory Efficiency**: Allows aggressive memory reuse (inplace ops) while keeping a "safety net" (snapshots) only when strictly necessary.
*   **Thread Safety**: All global maps (`g_snapshots`, `g_meta`) are protected by a global `std::mutex`, making the *metadata* management thread-safe.
*   **Alias Awareness**: Explicitly handles shared memory views. If you recompute one view, all other views are automatically updated.

### Cons & Potential Drawbacks
*   **Global Lock Contention**: The single global `mutex` (`g_lock`) guards all inplace operations for *all* graphs. in a highly multi-threaded environment (e.g., data parallel training with threads), this could become a bottleneck.
*   **Memory Overhead**: `g_snapshots` stores full tensor copies. If many large tensors are modified in-place, memory usage can spike (doubling the storage for those tensors).
*   **Complexity**: The interaction between "stale snapshots", "version counters", and "recursive recomputation" is complex. Debugging why a specific value is missing or wrong can be difficult.
*   **Memory Leaks (Global Maps)**: The global `g_snapshots` and `g_meta` maps hold pointers/references. If `clear_inplace_checkpoints()` is not called after graph execution finishes, these maps could grow indefinitely (memory leak) across multiple training "sessions".
*   **Snapshot Granularity**: Currently `store_delta` option exists in the struct but code seems to clone full tensors (`node->value.clone()`). Implementing true difference-based snapshots could save more memory.
