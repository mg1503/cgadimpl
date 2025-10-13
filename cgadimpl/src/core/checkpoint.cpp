// src/core/checkpoint.cpp
// Core checkpointing logic (manual + auto) for cgadimpl
// - mark_node_checkpoint: mark node and save minimal inputs (Value objects)
// - recompute_subgraph: restore inputs and re-run forward-eval for the node
// - simple auto_checkpoint_every_n utility

#include "ad/checkpoint.hpp"
#include <unordered_set>
#include <stdexcept>
#include <iostream>
#include <deque>
#include <queue>

namespace ag {
namespace checkpoint_impl {

using NodePtr = std::shared_ptr<Node>;

// For low-coupling we store RNG state as a small opaque blob.
// You can replace this with a repo-specific RNG state type.
using RngBlob = std::vector<uint8_t>;

// Save RNG state (stub). Extend to capture your RNG implementation.
static RngBlob save_rng_state() {
    RngBlob b;
    // TODO: capture RNG state (seed/counters) into blob
    return b;
}
static void restore_rng_state(const RngBlob &b) {
    (void)b;
    // TODO: restore RNG state from blob
}

// Mark node as checkpoint boundary and save minimal inputs.
void mark_node_checkpoint(const NodePtr &node, const CheckpointOptions &opts) {
    if (!node) return;
    if (node->is_checkpoint) return; // idempotent

    node->is_checkpoint = true;

    // Save minimal inputs as Values (these reference parents' nodes).
    node->saved_inputs.clear();
    for (auto &p : node->inputs) {
        if (p) node->saved_inputs.emplace_back(Value(p));
        else node->saved_inputs.emplace_back(Value()); // empty
    }

    if (opts.save_rng) {
        node->saved_rng_blob = save_rng_state();
        node->has_saved_rng = true;
    } else {
        node->has_saved_rng = false;
    }
}

// Recompute a checkpointed node: restore parents' saved tensors and call forward_eval_node
bool recompute_subgraph(const std::shared_ptr<Node>& node) {
    if (!node) return false;
    if (!node->is_checkpoint) return false;

    // We require that saved_inputs was populated when checkpoint was marked.
    if (node->saved_inputs.empty()) {
        std::cerr << "[checkpoint] no saved inputs for recompute\n";
        return false;
    }

    // Restore RNG if present
    if (node->has_saved_rng) {
        restore_rng_state(node->saved_rng_blob);
    }

    for (size_t i = 0; i < node->saved_inputs.size() && i < node->inputs.size(); ++i) {
        const Value &sv = node->saved_inputs[i];
        auto parent = node->inputs[i];
        if (!parent) continue;
        if (sv.node) {
            // Copy saved tensor into parent->value (shallow copy of Tensor object)
            parent->value = sv.node->value;
        } else {
            // If saved_value is empty but parent->value is missing, we may need to recompute parent.
            if (parent->value.size() == 0) {
                if (parent->is_checkpoint) {
                    // recursive recompute
                    if (!recompute_subgraph(parent)) {
                        std::cerr << "[checkpoint] failed to recompute parent\n";
                        return false;
                    }
                } else {
                    std::cerr << "[checkpoint] missing parent value and parent isn't checkpointed\n";
                    return false;
                }
            }
        }
    }

    // Now run forward evaluation for this node (must set node->value)
    try {
        Tensor out = forward_eval_node(node.get());
        node->value = out;
    } catch (const std::exception &e) {
        std::cerr << "[checkpoint] recompute exception: " << e.what() << "\n";
        return false;
    }
    return true;
}

// Helper: if node->is_checkpoint && node->value is empty, call recompute_subgraph
inline bool ensure_value_present(const NodePtr &node) {
    if (!node) return false;
    if (node->value.size() != 0) return true;
    if (node->is_checkpoint) return recompute_subgraph(node);
    return false;
}

// Simple auto-checkpoint: traverse nodes reachable from 'root' (BFS) and mark every n-th node
// Simple auto-checkpoint utility — declared in checkpoint.hpp


inline bool is_checkpointed(const NodePtr &node) {
    return node && node->is_checkpoint;
}


} // namespace checkpoint_impl
void auto_checkpoint_every_n(const Value &root, int n) {
    if (n <= 0 || !root.node) return;

    std::unordered_set<Node*> visited;
    std::deque<std::shared_ptr<Node>> q;
    q.push_back(root.node);
    int counter = 0;

    while (!q.empty()) {
        auto cur = q.front();
        q.pop_front();
        if (!cur || visited.count(cur.get())) continue;
        visited.insert(cur.get());

        // Mark every nth node as checkpoint
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
        auto [cur, depth] = q.front();
        q.pop();

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
