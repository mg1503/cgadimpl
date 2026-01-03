# Careful Deletion Documentation

This document provides a detailed overview of the Careful Deletion subsystem within the `cgadimpl` codebase. It outlines the namespaces, functions, functionality, dependencies, and an analysis of the pros and cons of the current implementation.

## Overview

The `careful_deletion` module allows for safe, dynamic memory management of the computation graph. In deep learning frameworks, intermediate tensors can consume vast amounts of memory. This system allows reusing or freeing that memory when it is no longer needed (e.g., after its contribution to the gradient is computed), while strictly respecting graph dependencies, checkpointing boundaries, and inplace operation aliases.

## Namespaces Used

*   **`ag`**: The primary namespace for the Autodiff Graph library.
*   **`ag::memory`**: A sub-namespace specifically for memory management utilities, including deletion policies and sweep functions.

## Dependencies

The `careful_deletion` module relies on the following internal and external components:

| File / Component | Type | Description |
| :--- | :--- | :--- |
| `<unordered_set>` | Standard Library | Used for `DeletionGuard` and passing sets of protected nodes. |
| `<algorithm>` | Standard Library | Used for sorting candidates in `sweep_with_checkpoint_priority`. |
| `<iostream>` | Standard Library | Used for printing debug statistics and memory pressure warnings. |
| `ad/core/graph.hpp` | Internal | **Crucial Dependency**. Defines `Node`, `Value`, `Op` (Leaf), and `topo_from`. |
| `ad/autodiff/inplace.hpp` | Internal | Used to check for active aliases (`has_alias`) and manage snapshots (`erase_snapshot`, `cleanup_stale_snapshots`). |
| `ad/autodiff/checkpoint.hpp` | Internal | Implicit dependency via `node->is_checkpoint` logic to prevent deleting recomputation anchors. |
| `ad/utils/debug.hpp` | Internal | General debugging utilities. |

## Functions Declared

The following functions are declared in `include/ad/autodiff/careful_deletion.hpp`:

```cpp
namespace ag {
namespace memory {

    enum class DeletePolicy { AlwaysSafe, Aggressive, ForwardPass };

    struct DeletionGuard {
        std::unordered_set<Node*> protected_nodes;
    };

    // Attempts to delete a single node's data if safe.
    bool try_delete_node(Node* node, DeletePolicy policy = DeletePolicy::AlwaysSafe, const std::unordered_set<Node*>* protected_nodes = nullptr);

    // Performs a full graph sweep to delete safe nodes.
    void sweep_safe_nodes(const Value& root, DeletePolicy policy = DeletePolicy::AlwaysSafe, const std::unordered_set<Node*>& protected_nodes = {});

    // Performs a memory-target-driven sweep, prioritizing non-checkpoints.
    void sweep_with_checkpoint_priority(const Value& root, size_t target_memory_mb);

    // Debugging and Stats
    void debug_deletion_state();
    void reset_deletion_stats();

} // namespace memory
} // namespace ag
```

## Functionality Breakdown

| Function | Functionality | Dependencies |
| :--- | :--- | :--- |
| **`try_delete_node`** | The core logic for safe deletion. <br> 1. **Checks**: Verifies if node is protected, a leaf, a checkpoint (unless policy overrides), has active aliases, or still needs gradients. <br> 2. **Memory Release**: If safe, clears `node->value`. Clears `node->grad` unless in `ForwardPass` mode (where grads are needed for accumulation). <br> 3. **Aggressive**: If `Aggressive` policy, also removes inplace snapshots. | `inplace::detail::has_alias`, `inplace::detail::erase_snapshot`, `Node::value/grad` |
| **`sweep_safe_nodes`** | Performs a garbage collection pass over the entire graph. <br> 1. **Cleanup**: Calls `inplace::cleanup_stale_snapshots()`. <br> 2. **Pressure Check**: Detects memory pressure (LOW/MED/HIGH) based on snapshot usage. Upgrades policy to `Aggressive` if pressure is HIGH. <br> 3. **Traversal**: Iterates via `topo_from`, calling `try_delete_node` on every node (except root). | `topo_from`, `try_delete_node`, `inplace::cleanup_stale_snapshots` |
| **`sweep_with_checkpoint_priority`** | A targeted cleanup to reach a specific memory budget (`target_memory_mb`). <br> 1. **Candidate Selection**: Collects all candidates that are finished with gradients and not aliased. <br> 2. **Sorting**: Sorts candidates to prioritize **non-checkpoint** nodes first, then largest memory usage. <br> 3. **Deletion**: Deletes nodes until the estimated freed memory satisfies the target. | `topo_from`, `estimate_node_memory`, `try_delete_node` |
| **`debug_deletion_state`** | Prints global statistics about the deletion subsystem: nodes deleted, nodes skipped, total memory freed, current memory pressure level, and snapshot memory usage. | `g_deletion_stats` (static global), `inplace::get_snapshot_memory_usage` |

## Analysis: Pros and Cons

### Pros
*   **Safety First**: The system is designed to be "Always Safe" by default. It explicitly checks for all conditions that could cause a crash (aliases, checkpoints, required gradients) before freeing anything.
*   **Adaptive**: The `sweep_safe_nodes` function automatically detects memory pressure and switches to an `Aggressive` policy if needed. This makes it robust for varying workload sizes.
*   **Granular Control**: Supports different `DeletePolicy` modes (`ForwardPass` for checkpointing, `Aggressive` for emergencies) allowing the runtime to tune behavior precisely.
*   **Targeted Cleanup**: `sweep_with_checkpoint_priority` is a sophisticated feature that allows the system to "make room" for new tensors by deleting the most "expensive" but "safe" old tensors, prioritizing those that don't trigger recomputation (non-checkpoints).

### Cons & Potential Drawbacks
*   **Global State**: The statistics (`g_deletion_stats`) are stored in a static global variable. This is not thread-safe if multiple independent graphs are being trained in parallel threads within the same process.
*   **Heuristic Thresholds**: The memory pressure detection uses hardcoded constants (1GB for HIGH pressure). This might be too small for high-end GPUs or too large for embedded devices. It lacks configuration options.
*   **Traversal Overhead**: `sweep_safe_nodes` calls `topo_from`, which traverses the entire graph. Doing this frequently (e.g., every iteration) could add CPU overhead, especially for graphs with millions of nodes.
*   **Memory Fragmentation**: "Deleting" a tensor usually means clearing its standard vector/storage. Depending on the underlying allocator, this might leave fragmented memory that isn't immediately reusable by the OS, though it is reusable by the process.
*   **Dependency Coupling**: Strongly coupled with `inplace` and `checkpoint` systems. Changes to how aliases work would require careful updates here to prevent use-after-free bugs.
