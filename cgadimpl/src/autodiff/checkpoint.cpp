//============================================================
// file: cgadimpl/src/core/checkpoint.cpp
//============================================================

#include "ad/autodiff/checkpoint.hpp"
#include <unordered_set>
#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include <deque>
#include <queue>
#include <algorithm>
#include <cmath>
#include "ad/autodiff/inplace.hpp"
#include "ad/ops/ops.hpp"

namespace ag {
namespace checkpoint_impl {

// ============================================================================
// Statistics and Monitoring
// ============================================================================

struct CheckpointStats {
    size_t total_checkpoints = 0;
    size_t successful_recomputes = 0;
    size_t failed_recomputes = 0;
    size_t total_memory_saved = 0;  // Estimated bytes saved
    size_t recompute_calls = 0;
    
    void reset() {
        total_checkpoints = 0;
        successful_recomputes = 0;
        failed_recomputes = 0;
        total_memory_saved = 0;
        recompute_calls = 0;
    }
    
    void print() const {
        std::cout << "=== Checkpoint Statistics ===\n";
        std::cout << "Total checkpoints: " << total_checkpoints << "\n";
        std::cout << "Successful recomputes: " << successful_recomputes << "\n";
        std::cout << "Failed recomputes: " << failed_recomputes << "\n";
        std::cout << "Estimated memory saved: " << (total_memory_saved / 1024.0 / 1024.0) << " MB\n";
        std::cout << "Recompute calls: " << recompute_calls << "\n";
        if (recompute_calls > 0) {
            std::cout << "Success rate: " << (100.0 * successful_recomputes / recompute_calls) << "%\n";
        }
        std::cout << "=============================\n";
    }
};

static CheckpointStats g_stats;

// ============================================================================
// Memory Estimation
// ============================================================================

static size_t estimate_tensor_memory(const Tensor& t) {
    if (t.numel() == 0) return 0;
    // Rough estimate: numel * sizeof(float) for typical tensor
    // Adjust based on actual dtype if available
    return t.numel() * sizeof(float);
}

static size_t estimate_node_memory(const std::shared_ptr<Node>& node) {
    if (!node) return 0;
    size_t total = 0;
    total += estimate_tensor_memory(node->value);
    total += estimate_tensor_memory(node->grad);
    return total;
}

// ============================================================================
// Smart Checkpoint Selection Policies
// ============================================================================

enum class CheckpointStrategy {
    EVERY_N,           // Checkpoint every N nodes
    BY_DEPTH,          // Checkpoint by depth threshold
    MEMORY_OPTIMAL,    // Maximize memory savings
    SPEED_OPTIMAL,     // Minimize recomputation overhead
    AUTO               // Automatic policy selection
};

struct NodeInfo {
    std::shared_ptr<Node> node;
    int depth;
    size_t memory_size;
    int num_children;
    int num_descendants;
};

// Calculate node importance for checkpointing
static double calculate_checkpoint_score(const NodeInfo& info, CheckpointStrategy strategy) {
    switch (strategy) {
        case CheckpointStrategy::MEMORY_OPTIMAL:
            // Prefer nodes with large memory and many descendants (high reuse)
            return static_cast<double>(info.memory_size) * std::log(1 + info.num_descendants);
        
        case CheckpointStrategy::SPEED_OPTIMAL:
            // Prefer nodes with fewer descendants (less recomputation)
            return static_cast<double>(info.memory_size) / std::max(1, info.num_descendants);
        
        case CheckpointStrategy::AUTO:
            // Balanced approach
            return static_cast<double>(info.memory_size) * std::sqrt(1 + info.num_descendants);
        
        default:
            return 0.0;
    }
}

// ============================================================================
// Forward Declarations
// ============================================================================

bool recompute_subgraph(const std::shared_ptr<Node>& node);
inline bool ensure_value_present(const std::shared_ptr<Node>& node);

// ============================================================================
// Core Checkpoint Marking
// ============================================================================

void mark_node_checkpoint(const std::shared_ptr<Node>& node, const CheckpointOptions& opts) {
    if (!node || node->is_checkpoint) return;
    
    node->is_checkpoint = true;
    node->saved_inputs.clear();
    
    // Save input references for recomputation
    for (auto& p : node->inputs) {
        node->saved_inputs.emplace_back(p ? Value(p) : Value());
    }
    
    // Track memory savings
    size_t memory_saved = estimate_node_memory(node);
    g_stats.total_memory_saved += memory_saved;
    g_stats.total_checkpoints++;
    
    if (opts.verbose) {
        std::cout << "[checkpoint] Marked node @" << node.get() 
                  << " (estimated " << (memory_saved / 1024.0) << " KB)\n";
    }
}

// ============================================================================
// Robust Recomputation Logic
// ============================================================================

inline bool ensure_value_present(const std::shared_ptr<Node>& node) {
    if (!node) return true;
    if (node->value.numel() != 0) return true;
    if (node->is_checkpoint) return recompute_subgraph(node);
    return false;
}

bool recompute_subgraph(const std::shared_ptr<Node>& node) {
    // std::cout << "[checkpoint] recompute_subgraph called for node @" << node.get() << "\n";
    if (!node) {
        g_stats.failed_recomputes++;
        return false;
    }
    
    if (!node->is_checkpoint) {
        g_stats.failed_recomputes++;
        return false;
    }
    
    g_stats.recompute_calls++;
    
    // Fast path: already recomputed
    // Check allocated_bytes() because reset() creates a scalar (numel=1) with 0 bytes
    // ALSO check if dims are empty. Default constructor creates scalar with empty dims.
    // We treat such tensors as "missing" (deleted) for checkpointing purposes.
    if (node->value.numel() != 0 && node->value.allocated_bytes() > 0 && !node->value.shape().dims.empty()) {
        return true;
    }
    
    // Recursively ensure all parents have values
    // Use saved_inputs as the source of truth for checkpointed nodes
    for (const auto& input_val : node->saved_inputs) {
        const auto& parent_node = input_val.node;
        if (!parent_node) continue;
        
        if (parent_node->value.numel() == 0 || parent_node->value.allocated_bytes() == 0) {
            if (parent_node->is_checkpoint) {
                if (!recompute_subgraph(parent_node)) {
                    std::cerr << "[checkpoint] ERROR: Failed to recompute parent @" 
                              << parent_node.get() << "\n";
                    g_stats.failed_recomputes++;
                    return false;
                }
            } else {
                std::cerr << "[checkpoint] ERROR: Non-checkpoint parent @" 
                          << parent_node.get() << " has no value\n";
                g_stats.failed_recomputes++;
                return false;
            }
        }
    }
    
    // Recompute this node
    try {
        // Restore inputs from saved_inputs to ensure graph connectivity
        if (node->inputs.size() != node->saved_inputs.size()) {
            node->inputs.resize(node->saved_inputs.size());
        }
        for (size_t i = 0; i < node->saved_inputs.size(); ++i) {
            node->inputs[i] = node->saved_inputs[i].node;
            if (!node->inputs[i]) {
                throw std::runtime_error("Restored input is null");
            }
        }

        node->value = forward_eval_node(node.get());
        ag::inplace::on_recomputed(node.get());
        g_stats.successful_recomputes++;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[checkpoint] Exception during recompute @" << node.get() 
                  << ": " << e.what() << "\n";
        g_stats.failed_recomputes++;
        return false;
    }
}

inline bool is_checkpointed(const std::shared_ptr<Node>& node) {
    return node && node->is_checkpoint;
}

// ============================================================================
// Smart Checkpoint Selection
// ============================================================================

void auto_checkpoint_smart(const Value& root, CheckpointStrategy strategy, double checkpoint_ratio = 0.2) {
    if (!root.node) return;
    
    // Gather node information
    std::unordered_map<Node*, NodeInfo> node_info_map;
    std::unordered_map<Node*, int> descendant_count;
    std::deque<std::shared_ptr<Node>> q;
    std::unordered_set<Node*> visited;
    
    q.push_back(root.node);
    int max_depth = 0;
    
    // First pass: collect nodes and calculate depths
    while (!q.empty()) {
        auto cur = q.front();
        q.pop_front();
        if (!cur || visited.count(cur.get())) continue;
        visited.insert(cur.get());
        
        NodeInfo info;
        info.node = cur;
        info.memory_size = estimate_node_memory(cur);
        info.num_children = cur->inputs.size();
        
        node_info_map[cur.get()] = info;
        
        for (auto& p : cur->inputs) {
            if (p) q.push_back(p);
        }
    }
    
    // Second pass: calculate descendant counts (bottom-up)
    auto topo_order = topo_from(root.node.get());
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        Node* n = *it;
        int count = 1;
        for (auto& child : n->inputs) {
            if (child) {
                count += descendant_count[child.get()];
            }
        }
        descendant_count[n] = count;
        if (node_info_map.count(n)) {
            node_info_map[n].num_descendants = count;
        }
    }
    
    // Calculate scores and select top candidates
    std::vector<std::pair<double, Node*>> scored_nodes;
    for (auto& [node_ptr, info] : node_info_map) {
        if (info.node->inputs.empty()) continue; // Skip leaf nodes
        double score = calculate_checkpoint_score(info, strategy);
        scored_nodes.push_back({score, node_ptr});
    }
    
    // Sort by score (descending)
    std::sort(scored_nodes.begin(), scored_nodes.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Checkpoint top N% of nodes
    size_t num_to_checkpoint = static_cast<size_t>(scored_nodes.size() * checkpoint_ratio);
    num_to_checkpoint = std::max(size_t(1), std::min(num_to_checkpoint, scored_nodes.size()));
    
    for (size_t i = 0; i < num_to_checkpoint; ++i) {
        Node* node_ptr = scored_nodes[i].second;
        if (node_info_map.count(node_ptr)) {
            mark_node_checkpoint(node_info_map[node_ptr].node, CheckpointOptions());
        }
    }
}

} // namespace checkpoint_impl

// ============================================================================
// Public API
// ============================================================================

void auto_checkpoint_every_n(const Value& root, int n) {
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
        ++counter;
        
        if (counter % n == 0 && !cur->inputs.empty()) {
            checkpoint_impl::mark_node_checkpoint(cur, CheckpointOptions());
        }
        
        for (auto& p : cur->inputs) {
            if (p) q.push_back(p);
        }
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
        
        for (auto& p : cur->inputs) {
            if (p) q.push({p, depth + 1});
        }
    }
}

// New smart checkpoint API
void auto_checkpoint_memory_optimal(const Value& root, double ratio) {
    checkpoint_impl::auto_checkpoint_smart(root, checkpoint_impl::CheckpointStrategy::MEMORY_OPTIMAL, ratio);
}

void auto_checkpoint_speed_optimal(const Value& root, double ratio) {
    checkpoint_impl::auto_checkpoint_smart(root, checkpoint_impl::CheckpointStrategy::SPEED_OPTIMAL, ratio);
}

void auto_checkpoint_balanced(const Value& root, double ratio) {
    checkpoint_impl::auto_checkpoint_smart(root, checkpoint_impl::CheckpointStrategy::AUTO, ratio);
}

void print_checkpoint_stats() {
    checkpoint_impl::g_stats.print();
}

void reset_checkpoint_stats() {
    checkpoint_impl::g_stats.reset();
}

} // namespace ag