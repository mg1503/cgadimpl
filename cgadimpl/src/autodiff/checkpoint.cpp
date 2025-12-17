//============================================================
// file: cgadimpl/src/core/checkpoint.cpp
//============================================================

#include "ad/autodiff/checkpoint.hpp"
#include <unordered_set>
#include <stdexcept>
#include <iostream>
#include <deque>
#include <queue>
#include "ad/autodiff/inplace.hpp"
#include "ad/ops/ops.hpp" // Required for forward_eval_node

namespace ag {
namespace checkpoint_impl {

// --- FORWARD DECLARATIONS to solve C++ ordering issues ---
bool recompute_subgraph(const std::shared_ptr<Node>& node);
inline bool ensure_value_present(const std::shared_ptr<Node>& node);

// ... (Your existing RNG stubs can go here if you have them) ...

void mark_node_checkpoint(const std::shared_ptr<Node> &node, const CheckpointOptions &opts) {
    if (!node || node->is_checkpoint) return;
    node->is_checkpoint = true;
    node->saved_inputs.clear();
    for (auto &p : node->inputs) {
        node->saved_inputs.emplace_back(p ? Value(p) : Value());
    }
    // ... (Your RNG saving logic can go here) ...
}

// --- HELPER FUNCTION that drives the robust recursion ---
inline bool ensure_value_present(const std::shared_ptr<Node> &node) {
    if (!node) return true; // Non-existent parents are not an error
    if (node->value.numel() != 0) return true; // Value is already present
    if (node->is_checkpoint) return recompute_subgraph(node); // Recompute it
    // If value is missing and not a checkpoint, we cannot proceed.
    return false; 
}

// --- ROBUST RECOMPUTATION LOGIC ---
bool recompute_subgraph(const std::shared_ptr<Node>& node) {
    if (!node) return false;
    if (!node->is_checkpoint) return false;

    // Fast path: if already recomputed by a recursive call, do nothing.
    if (node->value.numel() != 0) return true;

    // ... (Your RNG restore logic can go here) ...

    // 1. Recursively ensure all parents have their values present BEFORE we compute this node.
    for (const auto& parent_node : node->inputs) {
        if (!parent_node) continue;

        // Check if the parent's value is missing
        if (parent_node->value.numel() == 0) {
            // If the parent is also a checkpoint, we can recursively recompute it.
            if (parent_node->is_checkpoint) {
                if (!recompute_subgraph(parent_node)) {
                    std::cerr << "[checkpoint] ERROR: Failed to recursively recompute parent node @" 
                              << parent_node.get() << std::endl;
                    return false; // Propagate the failure up the chain
                }
            } else {
                // This is a critical failure. The parent's value is gone, and it's not a
                // checkpoint, so there is no way to bring it back.
                std::cerr << "[checkpoint] ERROR: Cannot recompute node @" << node.get()
                          << " because its non-checkpoint parent @" << parent_node.get() 
                          << " has been deallocated." << std::endl;
                return false;
            }
        }
    }

    // 2. Now that inputs are guaranteed to be present, run the forward op for THIS node.
    try {
        node->value = forward_eval_node(node.get());
        ag::inplace::on_recomputed(node.get());
    } catch (const std::exception &e) {
        std::cerr << "[checkpoint] Exception during recompute of node @" << node.get() << ": " << e.what() << "\n";
        return false;
    }

    return true;
}

inline bool is_checkpointed(const std::shared_ptr<Node> &node) {
    return node && node->is_checkpoint;
}

} // namespace checkpoint_impl

// --- Your existing auto_checkpoint functions ---

void auto_checkpoint_every_n(const Value &root, int n) {
    if (n <= 0 || !root.node) return;
    std::unordered_set<Node*> visited;
    std::deque<std::shared_ptr<Node>> q;
    q.push_back(root.node);
    int counter = 0;
    while (!q.empty()) {
        auto cur = q.front(); q.pop_front();
        if (!cur || visited.count(cur.get())) continue;
        visited.insert(cur.get());
        ++counter;
        if (counter % n == 0 && !cur->inputs.empty()) {
            checkpoint_impl::mark_node_checkpoint(cur, CheckpointOptions());
        }
        for (auto &p : cur->inputs)
            if (p) q.push_back(p);
    }
}

void auto_checkpoint_by_depth(const Value& root, int depth_threshold) {
    if (!root.node) return;
    struct QItem { std::shared_ptr<Node> node; int depth; };
    std::queue<QItem> q;
    std::unordered_set<Node*> visited;
    q.push({root.node, 0});
    while (!q.empty()) {
        auto [cur, depth] = q.front(); q.pop();
        if (!cur || visited.count(cur.get())) continue;
        visited.insert(cur.get());
        if (depth >= depth_threshold && !cur->inputs.empty()) {
            checkpoint_impl::mark_node_checkpoint(cur, CheckpointOptions());
        }
        for (auto &p : cur->inputs)
            if (p) q.push({p, depth + 1});
    }
}

} // namespace ag