// ===========================
// File: src/careful_delete.cpp
// ===========================
//
// Implements safe memory cleanup logic for autodiff nodes,
// respecting checkpointing, alias tracking, and gradient dependencies.
//

#include "ad/careful_deletion.hpp"
#include "ad/inplace.hpp"
#include "ad/debug.hpp"
#include <iostream>
#include <mutex>

namespace ag {
namespace memory {

using namespace ag::inplace;

// -----------------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------------

// Detect if node participates in an alias group (shares data buffer with others)
static bool has_active_alias(Node* n) {
    if (!n) return false;
    return inplace::detail::has_alias(n);
}

// Determine if all parent gradients have been propagated
static bool gradients_done(Node* n) {
    if (!n) return false;
    for (auto& p : n->inputs) {
        if (p && p->requires_grad)
            return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

bool try_delete_node(Node* node, DeletePolicy policy) {
    if (!node) return false;

    // 1️⃣ Skip parameter or constant leaves
    if (node->op == Op::Leaf) return false;

    // 2️⃣ Skip checkpoints (they are recomputed later)
    if (node->is_checkpoint) return false;

    // 3️⃣ Skip aliased tensors (shared storage)
    if (has_active_alias(node)) return false;

    // 4️⃣ Skip if gradient dependencies not yet resolved
    if (!gradients_done(node)) return false;

    // 5️⃣ Otherwise safe to free tensor memory
    node->value = Tensor();
    node->grad  = Tensor();

    // Optionally erase snapshot metadata for aggressive cleanup
    if (policy == DeletePolicy::Aggressive) {
        inplace::detail::erase_snapshot(node);
    }

    std::cout << "[careful_delete] Freed node@" << node
              << " op=" << op_name(node->op)
              << " policy=" << (policy==DeletePolicy::AlwaysSafe ? "Safe" : "Aggressive")
              << "\n";
    return true;
}

// Sweep over the entire graph and free safe nodes
void sweep_safe_nodes(const Value& root, DeletePolicy policy) {
    auto order = topo_from(root.node.get());
    int freed = 0;
    for (Node* n : order) {
        if (try_delete_node(n, policy))
            ++freed;
    }
    std::cout << "[careful_delete] Sweep complete. Freed " << freed << " nodes.\n";
}

// Debug utility
void debug_deletion_state() {
    std::cout << "=== Careful Deletion Debug ===\n";
    std::cout << "(This function prints only runtime-safe deletes)\n";
    // Could be extended later to show remaining node count, memory stats, etc.
}

} // namespace memory
} // namespace ag
