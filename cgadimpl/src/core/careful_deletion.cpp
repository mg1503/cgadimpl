// ===========================
// File: src/careful_delete.cpp
// ===========================
//
// Implements safe memory cleanup logic for autodiff nodes,
// respecting checkpointing, alias tracking, and gradient dependencies.
//
// The functions in this file are part of the `ag::memory` subsystem
// and are designed to help reclaim memory safely during or after
// forward/backward computations without breaking the computational graph.
//
// They ensure that tensors are only freed when they are no longer needed
// for gradient computation, recomputation, or alias/view consistency.
//

#include "ad/careful_deletion.hpp"
#include "ad/inplace.hpp"
#include "ad/debug.hpp"
#include <iostream>
#include <mutex>

namespace ag {
namespace memory {

using namespace ag::inplace;

/*
 * =============================================================
 * Overview
 * =============================================================
 * 
 * The goal of this file is to manage **safe tensor deletion** for graph nodes
 * while maintaining correctness in automatic differentiation (autodiff).
 * 
 * In deep learning frameworks, each operation (node) holds:
 *     - a `value` tensor (its output),
 *     - a `grad` tensor (for accumulated gradient),
 *     - connections to its input nodes (parents).
 *
 * After a backward pass, many of these tensors are no longer needed.
 * However, we cannot simply free them immediately, because:
 *     - Some nodes are part of **checkpoints** (will be recomputed later).
 *     - Some tensors share memory with other tensors (aliases/views).
 *     - Some nodes’ gradients have not yet been fully propagated.
 *
 * To handle this safely, this file defines rules to determine when
 * a node’s data can be safely freed.
 *
 * The main functions are:
 *      `try_delete_node()` – checks and frees a single node safely.
 *      `sweep_safe_nodes()` – performs a global cleanup pass.
 *      `debug_deletion_state()` – prints the current deletion summary.
 */

// -----------------------------------------------------------------------------
// Internal helper functions (not exposed to users)
// -----------------------------------------------------------------------------

/*
 *  has_active_alias():
 *  -------------------
 *  Checks whether the given node participates in an alias group.
 *  Inplace or view operations can cause multiple tensors to share
 *  the same underlying data storage (e.g., slicing, transpose, etc.).
 *
 *  If such aliasing exists, deleting one tensor may accidentally
 *  invalidate others, so deletion must be skipped.
 *
 *  Uses `inplace::detail::has_alias()` — an internal helper that
 *  tracks alias groups created by in-place operations.
 *
 *  Parameters:
 *      n : pointer to a Node in the computational graph
 *
 *  Returns:
 *      true  — if this node’s tensor storage is shared with others
 *      false — if the tensor has exclusive ownership
 */
static bool has_active_alias(Node* n) {
    if (!n) return false;
    return inplace::detail::has_alias(n);
}

/*
 *  gradients_done():
 *  -----------------
 *  Checks whether all gradient dependencies for a node have been resolved.
 *  In autodiff, nodes accumulate gradients from their children.
 *  We must not delete a node until all its parents’ gradient computations
 *  are finished, or it could break the gradient chain.
 *
 *  This simple check iterates over the node’s input edges and ensures
 *  that no parent still requires gradient updates.
 *
 *  Parameters:
 *      n : pointer to a Node
 *
 *  Returns:
 *      true  — if no parent node still requires gradients
 *      false — if gradient propagation is still ongoing
 */
static bool gradients_done(Node* n) {
    if (!n) return false;
    for (auto& p : n->inputs) {
        if (p && p->requires_grad)
            return false;  // at least one parent still needs its gradient
    }
    return true;
}

// -----------------------------------------------------------------------------
// Public API: Core Safe Deletion Logic
// -----------------------------------------------------------------------------

/*
 *  try_delete_node():
 *  -------------------
 *  Safely attempts to free a single node’s tensor data (`value` and `grad`)
 *  based on several safety checks and a user-specified deletion policy.
 *
 *  This function performs a series of ordered checks to decide
 *  whether the node can be safely deleted:
 *
 *    1 Skip leaf nodes (`Op::Leaf`) — these are constants or parameters.
 *    2 Skip checkpointed nodes — their activations are needed for recomputation.
 *    3 Skip aliased tensors — shared buffers cannot be safely deleted.
 *    4 Skip nodes whose gradients are not yet fully propagated.
 *    5 If all checks pass → delete value and gradient tensors.
 *
 *  Parameters:
 *      node   : pointer to target Node to check and delete
 *      policy : deletion behavior
 *               - `AlwaysSafe` → only delete when 100% safe.
 *               - `Aggressive` → delete eagerly, even if risky.
 *
 *  Returns:
 *      true  — if the node’s data was safely freed.
 *      false — if the node was skipped for safety.
 *
 *  Notes:
 *      - This is designed to be called repeatedly after backward().
 *      - Aggressive deletion optionally erases in-place metadata via
 *        `inplace::detail::erase_snapshot(node)` to free alias info too.
 */
bool try_delete_node(Node* node, DeletePolicy policy) {
    if (!node) return false;

    // 1 Skip parameter or constant leaf nodes (never delete their tensors)
    if (node->op == Op::Leaf) return false;

    // 2  Skip checkpoint nodes — these must remain for recomputation
    if (node->is_checkpoint) return false;

    // 3  Skip nodes with active alias relationships
    if (has_active_alias(node)) return false;

    // 4  Skip if gradients are still required by parent nodes
    if (!gradients_done(node)) return false;

    // 5 Otherwise, it is safe to free this node’s memory
    node->value = Tensor();  // release the tensor’s data buffer
    node->grad  = Tensor();  // release gradient memory as well

    // Optional: if aggressive policy, clear alias or metadata info completely
    if (policy == DeletePolicy::Aggressive) {
        inplace::detail::erase_snapshot(node);
    }

    // Debug message — logs every node freed
    std::cout << "[careful_delete] Freed node@" << node
              << " op=" << op_name(node->op)
              << " policy=" << (policy == DeletePolicy::AlwaysSafe ? "Safe" : "Aggressive")
              << "\n";
    return true;
}

// -----------------------------------------------------------------------------
// sweep_safe_nodes()
// -----------------------------------------------------------------------------

/*
 *  sweep_safe_nodes():
 *  --------------------
 *  Traverses the entire computation graph starting from the given `root`
 *  and attempts to delete all nodes that can be safely freed according to
 *  the current policy.
 *
 *  Process:
 *      - Performs a topological traversal of all reachable nodes.
 *      - For each node, calls `try_delete_node(node, policy)`.
 *      - Tracks how many nodes were successfully freed.
 *
 *  Parameters:
 *      root   : The Value object (typically the model output or loss)
 *      policy : Deletion policy (Safe or Aggressive)
 *
 *  Output:
 *      Prints how many nodes were deleted during this cleanup pass.
 *
 *  Typical usage:
 *      after backward():
 *          memory::sweep_safe_nodes(output, memory::DeletePolicy::AlwaysSafe);
 */
void sweep_safe_nodes(const Value& root, DeletePolicy policy) {
    auto order = topo_from(root.node.get()); // Get nodes in topological order
    int freed = 0;
    for (Node* n : order) {
        if (try_delete_node(n, policy))
            ++freed;
    }
    std::cout << "[careful_delete] Sweep complete. Freed " << freed << " nodes.\n";
}

// -----------------------------------------------------------------------------
// debug_deletion_state()
// -----------------------------------------------------------------------------

/*
 *  debug_deletion_state():
 *  ------------------------
 *  Prints the current deletion system state for debugging purposes.
 *  Intended for developers to monitor how many deletions occurred
 *  and whether the policy is operating correctly.
 *
 *  Current output:
 *      - Header label
 *      - Static message (can be extended for memory stats)
 *
 *  Future extensions might include:
 *      - Number of live nodes
 *      - Number of protected (checkpointed / aliased) nodes
 *      - Approximate freed memory in bytes
 *      - Snapshot statistics from inplace::detail
 */
void debug_deletion_state() {
    std::cout << "=== Careful Deletion Debug ===\n";
    std::cout << "(This function prints only runtime-safe deletes)\n";
    // Future work: add statistics on remaining nodes or allocated bytes
}

} // namespace memory
} // namespace ag
