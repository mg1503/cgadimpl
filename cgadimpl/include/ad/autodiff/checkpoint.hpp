//============================================================
// file: cgadimpl/include/ad/checkpoint.hpp
//============================================================
#pragma once
#include <vector>
#include <memory>
#include <cstdint>
#include "ad/ops/ops.hpp"
#include "ad/core/graph.hpp"
#include "ad/core/schema.hpp"

namespace ag {

/*
 *  This header defines the interface and configuration for **activation checkpointing**
 *  — a technique used to trade computation for memory efficiency during backpropagation.
 *
 *  During training, we sometimes "checkpoint" certain nodes in the computational graph.
 *  Instead of storing all intermediate activations for the backward pass,
 *  we store only a few "checkpoint" activations and recompute others as needed.
 *
 *  Checkpointing helps reduce GPU/CPU memory usage during training of large models.
 *
 *  In this file:
 *    - `CheckpointOptions`: config structure for controlling checkpoint behavior.
 *    - `auto_checkpoint_every_n` / `auto_checkpoint_by_depth`: automatic checkpointing utilities.
 *    - `checkpoint_impl` namespace: low-level internal functions used to manage and recompute checkpoints.
 */

/*
 *  The `CheckpointOptions` structure defines various flags and parameters that
 *  control how checkpointing behaves for a node or subgraph.
 */
struct CheckpointOptions {
    bool save_rng = true;         // Whether to save RNG (random number generator) state.
                                 // Useful for deterministic recomputation, especially when dropout or random ops are used.

    int max_recompute_depth = 1000; // Limits how deep recursive recomputation is allowed to go.
                                    // Prevents infinite recursion or excessive recomputation in very deep graphs.

    bool save_inputs = true;      // If true, save the input Values (tensors) of this node during checkpointing.
                                 // This allows easy restoration of inputs during recomputation.

    bool detach_inputs = false;   // If true, inputs are detached from the computation graph during checkpointing,
                                 // breaking gradient connections. Used for special memory optimization cases.

    bool force = false;           // Forces checkpointing even when not strictly required.
                                 // Useful for debugging or manual override.

    bool verbose = false;         // If true, prints verbose logs during checkpoint marking and recomputation.
                                 // Helpful for debugging and tracing recompute behavior.
    
};

/*
 *  (Optional user-facing API, commented out in the source)
 *
 *  The checkpoint() function allows users to manually wrap a Value in a checkpoint.
 *  When backpropagation occurs, intermediate activations for this node will not be kept in memory,
 *  and the forward pass for this node will be recomputed when needed.
 *
 *  Example:
 *      Value y = checkpoint(forward_pass(x));
 *
 *  This helps reduce memory at the cost of additional computation during backward().
 */
// Value checkpoint(const Value &v, const CheckpointOptions &opts = CheckpointOptions());

/*
 *  Utility function 1:
 *  --------------------
 *  Automatically mark every Nth node in the computational graph (starting from `root`)
 *  as a checkpoint node. This helps distribute checkpoint nodes uniformly across the graph.
 *
 *  Parameters:
 *      - root: The root Value of the computation graph (usually model output).
 *      - n: Interval — every nth node is checkpointed.
 */
void auto_checkpoint_every_n(const Value &root, int n);

/*
 *  Utility function 2:
 *  --------------------
 *  Automatically mark all nodes beyond a certain depth threshold from the root as checkpoints.
 *  This is another automated policy where deeper nodes (which tend to hold larger activations)
 *  are more likely to be checkpointed.
 *
 *  Parameters:
 *      - root: Root of the computation graph.
 *      - depth_threshold: Depth (distance from root) after which nodes are checkpointed.
 */
void auto_checkpoint_by_depth(const Value& root, int depth_threshold);

/*
 *  Smart Checkpoint Selection APIs:
 *  ---------------------------------
 *  These functions use intelligent policies to select which nodes to checkpoint
 *  based on memory usage, recomputation cost, and graph structure.
 */

/*
 *  auto_checkpoint_memory_optimal():
 *  ---------------------------------
 *  Selects checkpoints to maximize memory savings.
 *  Prefers nodes with large memory footprint and high reuse (many descendants).
 *
 *  Parameters:
 *      - root: Root of the computation graph
 *      - ratio: Fraction of nodes to checkpoint (0.0 to 1.0, default 0.2 = 20%)
 */
void auto_checkpoint_memory_optimal(const Value& root, double ratio = 0.2);

/*
 *  auto_checkpoint_speed_optimal():
 *  --------------------------------
 *  Selects checkpoints to minimize recomputation overhead.
 *  Prefers nodes with large memory but fewer descendants (less recomputation).
 *
 *  Parameters:
 *      - root: Root of the computation graph
 *      - ratio: Fraction of nodes to checkpoint (0.0 to 1.0, default 0.2 = 20%)
 */
void auto_checkpoint_speed_optimal(const Value& root, double ratio = 0.2);

/*
 *  auto_checkpoint_balanced():
 *  ---------------------------
 *  Balanced checkpoint selection policy.
 *  Balances between memory savings and recomputation cost.
 *
 *  Parameters:
 *      - root: Root of the computation graph
 *      - ratio: Fraction of nodes to checkpoint (0.0 to 1.0, default 0.2 = 20%)
 */
void auto_checkpoint_balanced(const Value& root, double ratio = 0.2);

/*
 *  Statistics and Monitoring:
 *  --------------------------
 *  Functions to track and display checkpoint performance metrics.
 */

/*
 *  print_checkpoint_stats():
 *  -------------------------
 *  Prints current checkpoint statistics including:
 *  - Total checkpoints created
 *  - Successful/failed recomputes
 *  - Estimated memory saved
 *  - Recompute success rate
 */
void print_checkpoint_stats();

/*
 *  reset_checkpoint_stats():
 *  -------------------------
 *  Resets all checkpoint statistics to zero.
 *  Useful for benchmarking different checkpoint strategies.
 */
void reset_checkpoint_stats();

/*
 *  Internal namespace `checkpoint_impl`
 *  -------------------------------------
 *  These functions provide the internal mechanisms for:
 *    - marking nodes as checkpoints,
 *    - saving minimal input state,
 *    - restoring/recomputing node values when needed.
 *
 *  These are used internally by the autograd system and `backward()`.
 */
namespace checkpoint_impl {

/*
 *  mark_node_checkpoint():
 *  ------------------------
 *  Marks a given node as a checkpoint boundary.
 *  It performs the following:
 *    - Sets `node->is_checkpoint = true`.
 *    - Stores minimal input tensors (`Value` objects) into `node->saved_inputs`.
 *    - Optionally saves RNG (random state) into `node->saved_rng_blob`.
 *
 *  Inputs:
 *      - node: Shared pointer to a computational graph node.
 *      - opts: Configuration options for checkpointing (defaults are fine for most cases).
 */
void mark_node_checkpoint(const std::shared_ptr<Node> &node, const CheckpointOptions &opts = CheckpointOptions());

/*
 *  recompute_subgraph():
 *  ----------------------
 *  Recomputes the forward pass of a checkpointed node when its outputs
 *  are needed (for example, during backward() gradient propagation).
 *
 *  The function:
 *    1. Restores saved RNG state (if any).
 *    2. Restores or recursively recomputes inputs of the node.
 *    3. Calls `forward_eval_node(node.get())` to recompute the output tensor.
 *
 *  Returns:
 *      true  - if recomputation succeeded.
 *      false - if it failed due to missing data or dependency errors.
 */
bool recompute_subgraph(const std::shared_ptr<Node>& node);

/*
 *  is_checkpointed():
 *  ------------------
 *  Lightweight helper function that checks whether a given node
 *  has been marked as a checkpoint.
 *
 *  Returns true if node exists and node->is_checkpoint == true.
 */
inline bool is_checkpointed(const std::shared_ptr<Node> &node);

} // namespace checkpoint_impl

} // namespace ag