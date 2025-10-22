//============================================================
// file: cgadimpl/include/ad/careful_deletion.cpp
//============================================================
#pragma once
#include "ad/graph.hpp"
#include <unordered_set>

namespace ag {
namespace memory {

/*
 *  ============================================================
 *  Purpose:
 *  ============================================================
 *  This header defines an internal subsystem for **safe memory deletion** 
 *  of nodes and tensors in the computation graph.
 *
 *  During training or gradient computation, many intermediate tensors 
 *  (values and gradients) become no longer needed once their contribution 
 *  to the final result is complete.
 *
 *  However, in frameworks that support:
 *      - checkpointing (recompute needed nodes later),
 *      - shared views / aliasing,
 *      - and inplace operations,
 *  deleting tensors too early can corrupt the graph or crash the program.
 *
 *  Hence, this “careful deletion” mechanism allows memory cleanup 
 *  while respecting graph dependencies and checkpoint safety.
 *
 *  Key functionalities provided:
 *   - **Deletion policy control** (safe vs aggressive).
 *   - **Protection registry** (`DeletionGuard`) to track which nodes 
 *     are currently protected from deletion.
 *   - **try_delete_node()** for node-level cleanup.
 *   - **sweep_safe_nodes()** for graph-level cleanup passes.
 *   - **debug_deletion_state()** for monitoring what’s being deleted or protected.
 *
 *  This system integrates with checkpointing and autodiff to manage 
 *  dynamic memory efficiently without compromising correctness.
 */

// ------------------------------------------------------------
// DeletePolicy — controls how strict memory deletion should be
// ------------------------------------------------------------

/*
 *  enum class DeletePolicy:
 *  ------------------------
 *  Defines strategies that govern when a node’s memory (tensor or grad)
 *  can be safely released.
 *
 *  This allows flexible balancing between safety and memory efficiency.
 */
enum class DeletePolicy {
    AlwaysSafe,  // Only free tensors if we are certain no other part
                 // of the graph (e.g., a checkpoint or backward link)
                 // still depends on them.

    Aggressive,  // Forcefully delete tensors when not explicitly protected.
                 // Higher risk of recomputation errors if used carelessly,
                 // but saves more memory in long training runs.
};

// ------------------------------------------------------------
// DeletionGuard — registry for temporarily protected nodes
// ------------------------------------------------------------

/*
 *  DeletionGuard:
 *  ---------------
 *  A structure that tracks nodes which are currently “protected”
 *  from deletion. This is typically used when:
 *      - A node is part of an active checkpoint boundary
 *      - A backward pass is in progress
 *      - Lazy recomputation or alias tracking needs to preserve tensors
 *
 *  Implementation:
 *      - `protected_nodes` stores raw Node pointers.
 *      - Used as a lightweight runtime guard — not for ownership.
 */
struct DeletionGuard {
    std::unordered_set<Node*> protected_nodes;  // Active protection set
};

// ------------------------------------------------------------
// try_delete_node()
// ------------------------------------------------------------

/*
 *  try_delete_node():
 *  -------------------
 *  Attempts to delete (free) a node’s `value` tensor and/or its gradient.
 *
 *  Behavior:
 *      - Checks the node’s safety based on current deletion policy.
 *      - Ensures no checkpoint or guard is still referencing the node.
 *      - If safe → clears the Tensor contents (`node->value.clear()` etc.).
 *      - Returns `true` if deletion succeeded, `false` otherwise.
 *
 *  Parameters:
 *      node    : pointer to the target Node in the computation graph.
 *      policy  : deletion policy to use (default = AlwaysSafe).
 *
 *  Typical usage:
 *      if (memory::try_delete_node(n)) {
 *          std::cout << "Freed tensor for node " << n << std::endl;
 *      }
 *
 *  This function helps reclaim unused intermediate tensors 
 *  during forward/backward passes without corrupting graph state.
 */
bool try_delete_node(Node* node, DeletePolicy policy = DeletePolicy::AlwaysSafe);

// ------------------------------------------------------------
// sweep_safe_nodes()
// ------------------------------------------------------------

/*
 *  sweep_safe_nodes():
 *  --------------------
 *  Performs a full-graph traversal starting from `root` 
 *  and attempts to delete all nodes that are deemed safe
 *  according to the given deletion policy.
 *
 *  Implementation idea:
 *      1. Traverse the graph (BFS or topological order).
 *      2. For each node:
 *          - Check if it’s protected (checkpointed or guarded).
 *          - If not → call `try_delete_node(node)`.
 *
 *  Parameters:
 *      root    : The Value object representing the computation graph’s root.
 *      policy  : Determines how aggressively to delete nodes.
 *
 *  Example:
 *      memory::sweep_safe_nodes(output_value, memory::DeletePolicy::Aggressive);
 *
 *  This helps perform bulk cleanup after forward or backward passes.
 */
void sweep_safe_nodes(const Value& root, DeletePolicy policy = DeletePolicy::AlwaysSafe);

// ------------------------------------------------------------
// debug_deletion_state()
// ------------------------------------------------------------

/*
 *  debug_deletion_state():
 *  ------------------------
 *  Prints current memory management state to stdout.
 *
 *  Use this for debugging or profiling memory reuse behavior.
 *  Typically, it reports:
 *      - How many nodes are protected
 *      - Which nodes were deleted or skipped
 *      - Active policy in effect
 *
 *  This is especially useful when tuning deletion thresholds 
 *  in complex models (like large transformers).
 */
void debug_deletion_state();

} // namespace memory
} // namespace ag