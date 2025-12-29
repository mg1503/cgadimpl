# Checkpoint Documentation

This document provides a detailed overview of the Checkpoint subsystem within the `cgadimpl` codebase. It outlines the namespaces, functions, functionality, dependencies, and an analysis of the pros and cons of the current implementation.

## Overview

The `checkpoint` module enables **gradient checkpointing** (also known as activation checkpointing). This technique trades computation for memory by deleting intermediate activations during the forward pass and recomputing them on-demand during the backward pass. This is essential for training deep models (like Transformers) on hardware with limited VRAM.

## Namespaces Used

*   **`ag`**: The primary namespace for the Autodiff Graph library.
*   **`ag::checkpoint_impl`**: Internal namespace for core implementation details of marking and recomputing checkpoints.

## Dependencies

The `checkpoint` module relies on the following internal and external components:

| File / Component | Type | Description |
| :--- | :--- | :--- |
| `<vector>`, `<deque>`, `<queue>` | Standard Library | Used for graph traversal (BFS) and managing lists of saved inputs. |
| `<unordered_map>`, `<unordered_set>` | Standard Library | Used for graph traversal and storing node metadata. |
| `<algorithm>, <cmath>` | Standard Library | Used for scoring and sorting checkpoint candidates. |
| `ad/core/graph.hpp` | Internal | Defines `Node`, `Value`, and graph structure. |
| `ad/core/schema.hpp` | Internal | Likely defines `Tensor`, `Shape`, and other data types. |
| `ad/ops/ops.hpp` | Internal | Used to access `forward_eval_node` for recomputation. |
| `ad/autodiff/inplace.hpp` | Internal | Used to handle `inplace::on_recomputed` hooks. |

## Functions Declared

The following functions are declared in `include/ad/autodiff/checkpoint.hpp`:

```cpp
namespace ag {

    struct CheckpointOptions { ... };

    // Standard Policies
    void auto_checkpoint_every_n(const Value &root, int n);
    void auto_checkpoint_by_depth(const Value &root, int depth_threshold);

    // Smart Policies
    void auto_checkpoint_memory_optimal(const Value& root, double ratio = 0.2);
    void auto_checkpoint_speed_optimal(const Value& root, double ratio = 0.2);
    void auto_checkpoint_balanced(const Value& root, double ratio = 0.2);

    // Stats
    void print_checkpoint_stats();
    void reset_checkpoint_stats();

    namespace checkpoint_impl {
        void mark_node_checkpoint(const std::shared_ptr<Node> &node, const CheckpointOptions &opts = CheckpointOptions());
        bool recompute_subgraph(const std::shared_ptr<Node>& node);
        inline bool is_checkpointed(const std::shared_ptr<Node> &node);
    }
}
```

## Functionality Breakdown

| Function | Functionality | Dependencies |
| :--- | :--- | :--- |
| **`mark_node_checkpoint`** | Marks a node as a "restart point" for recomputation. <br> 1. Sets `is_checkpoint = true`. <br> 2. Saves lightweight references to input `Value`s (not the tensors themselves unless they are also checkpoints or leaves). <br> 3. Optionally saves RNG state for deterministic re-execution. | `Node::saved_inputs`, `Node::is_checkpoint` |
| **`recompute_subgraph`** | The engine of the system. Called during `backward()` when a value is missing. <br> 1. **Validity Check**: Returns immediately if data exists. <br> 2. **Recursive Restoration**: Recursively ensures all *inputs* to the current node are available (calling `recompute_subgraph` on parents if needed). <br> 3. **Execution**: Calls `forward_eval_node` to regenerate the `value` tensor. <br> 4. **Hooks**: Triggers `on_recomputed` for inplace version tracking. | `forward_eval_node` (`ops.hpp`), `inplace::on_recomputed` |
| **`auto_checkpoint_every_n`** | Simple heuristic: checkpoints every Nth node in a BFS traversal. Good for linear models; often suboptimal for complex DAGs. | BFS (`std::deque`) |
| **`auto_checkpoint_smart`** | **Internal helper** for optimal strategies. <br> 1. Analyzes the full graph to compute "scores" for each node based on memory footprint and descendant count. <br> 2. Sorts nodes by score. <br> 3. Checkpoints the top `ratio` (e.g., 20%) of nodes. | `estimate_node_memory`, `topo_from`, `calculate_checkpoint_score` |
| **`auto_checkpoint_...`** | Public wrappers for `auto_checkpoint_smart` using different scoring formulas: <br> - **Memory**: Prioritizes large tensors with many users. <br> - **Speed**: Prioritizes large tensors with few users (cheap to recompute). <br> - **Balanced**: Geometric mean of factors. | `auto_checkpoint_smart` |

## Analysis: Pros and Cons

### Pros
*   **Plug-and-Play Memory Savings**: Can drastically reduce memory usage (often by 3-4x) for large models with a single function call (`auto_checkpoint_balanced`).
*   **Smart Selection**: The "smart" strategies (Memory/Speed optimal) are sophisticated. They don't just guess; they analyze graph topology and tensor sizes to pick the *best* places to checkpoint.
*   **Deterministic**: Correctly handles RNG state (commented/implied in options), ensuring that recomputing a dropout layer produces the same mask as the first pass.
*   **Recursive Robustness**: `recompute_subgraph` is capable of handling chains of checkpoints (checkpoints depending on other checkpoints), reconstructing the graph state depth-first.

### Cons & Potential Drawbacks
*   **Recomputation Cost**: Checkpointing inevitably slows down training (typically 20-30%) because the forward pass for checkpointed segments runs twice.
*   **Complexity**: `recompute_subgraph` is recursive. In extremely deep chains of checkpoints, this could theoretically overflow the C++ stack (though `max_recompute_depth` option attempts to mitigate this).
*   **Coupling**: Heavily coupled with `ops.hpp` (`forward_eval_node`). Use of `shared_ptr` everywhere adds some overhead compared to raw pointer graphs, though it ensures safety.
*   **Global Stats**: Uses a static global `g_stats` for metrics. Not thread-safe for parallel model training in the same process.
*   **Memory Fragmentation**: Repeatedly allocating and freeing tensors during recompute cycles can cause memory fragmentation in the allocator if not handled by a specialized pool allocator (like `OwnTensor`).
