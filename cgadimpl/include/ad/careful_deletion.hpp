#pragma once
#include "ad/graph.hpp"
#include <unordered_set>

namespace ag {
namespace memory {

// Policy flags
enum class DeletePolicy {
    AlwaysSafe,      // Only free when definitely safe
    Aggressive,      // Force delete if no active checkpoint
};

// Tracks which nodes are currently protected
struct DeletionGuard {
    std::unordered_set<Node*> protected_nodes;
};

// Attempt to safely delete a node’s tensor value and/or grad
bool try_delete_node(Node* node, DeletePolicy policy = DeletePolicy::AlwaysSafe);

// Sweep through graph and delete safe nodes
void sweep_safe_nodes(const Value& root, DeletePolicy policy = DeletePolicy::AlwaysSafe);

// Print debug info
void debug_deletion_state();

} // namespace memory
} // namespace ag
