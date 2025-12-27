// ===========================
// File: cgadimpl/src/careful_delete.cpp
// ===========================
//
// Implements safe memory cleanup logic for autodiff nodes,
// respecting checkpointing, alias tracking, and gradient dependencies.
//
// Enhanced with checkpoint integration and memory pressure detection.

#include "ad/autodiff/careful_deletion.hpp"
#include "ad/autodiff/inplace.hpp"
#include "ad/autodiff/checkpoint.hpp"
#include "ad/utils/debug.hpp"
#include <iostream>
#include <mutex>
#include <algorithm>

namespace ag {
namespace memory {

using namespace ag::inplace;

// ============================================================================
// Statistics and Monitoring
// ============================================================================

struct DeletionStats {
    size_t nodes_deleted = 0;
    size_t nodes_skipped = 0;
    size_t memory_freed = 0;
    
    void reset() {
        nodes_deleted = 0;
        nodes_skipped = 0;
        memory_freed = 0;
    }
    
    void print() const {
        std::cout << "=== Memory Deletion Statistics ===\n";
        std::cout << "Nodes deleted: " << nodes_deleted << "\n";
        std::cout << "Nodes skipped: " << nodes_skipped << "\n";
        std::cout << "Memory freed: " << (memory_freed / 1024.0 / 1024.0) << " MB\n";
        std::cout << "==================================\n";
    }
};

static DeletionStats g_deletion_stats;

// ============================================================================
// Memory Estimation
// ============================================================================

static size_t estimate_node_memory(Node* node) {
    if (!node) return 0;
    size_t total = 0;
    if (node->value.numel() > 0) {
        total += node->value.numel() * sizeof(float);
    }
    if (node->grad.numel() > 0) {
        total += node->grad.numel() * sizeof(float);
    }
    return total;
}

// ============================================================================
// Helper Functions
// ============================================================================

static bool has_active_alias(Node* n) {
    if (!n) return false;
    return inplace::detail::has_alias(n);
}

static bool gradients_done(Node* n) {
    if (!n) return false;
    for (auto& p : n->inputs) {
        if (p && p->requires_grad())
            return false;
    }
    return true;
}

// ============================================================================
// Enhanced Deletion Logic
// ============================================================================

bool try_delete_node(Node* node, DeletePolicy policy, const std::unordered_set<Node*>* protected_nodes) {
    if (!node) return false;

    // 0. Check explicit protection
    if (protected_nodes && protected_nodes->count(node)) {
        g_deletion_stats.nodes_skipped++;
        return false;
    }

    // 1. Skip leaf nodes
    if (node->op == Op::Leaf) {
        g_deletion_stats.nodes_skipped++;
        return false;
    }

    // 2. Skip checkpoint nodes (unless aggressive policy or forward pass)
    // For ForwardPass, we allow deleting checkpoints because we use explicit protection
    // for the ones we want to keep (anchors).
    if (node->is_checkpoint && policy != DeletePolicy::Aggressive && policy != DeletePolicy::ForwardPass) {
        g_deletion_stats.nodes_skipped++;
        return false;
    }

    // 3. Skip nodes with active aliases
    if (has_active_alias(node)) {
        g_deletion_stats.nodes_skipped++;
        return false;
    }

    // 4. Skip if gradients still required
    // Exception: If policy is ForwardPass, we assume recomputation is possible,
    // so we can delete even if gradients are needed (as long as it's not a checkpoint).
    if (policy != DeletePolicy::ForwardPass && !gradients_done(node)) {
        g_deletion_stats.nodes_skipped++;
        return false;
    }

    // 5. Estimate memory to be freed
    size_t mem_freed = estimate_node_memory(node);

    // 6. Free memory
    // Explicitly assign empty tensor to ensure numel() == 0.
    node->value = Tensor();
    
    // Only delete gradient if NOT in ForwardPass mode.
    // In ForwardPass (checkpointing), we need the gradient tensor to remain valid (allocated by zero_grad)
    // because autodiff expects it to have correct shape for accumulation and VJP.
    if (policy != DeletePolicy::ForwardPass) {
        node->grad = Tensor();
    }

    // 7. Optional: aggressive cleanup
    if (policy == DeletePolicy::Aggressive) {
        inplace::detail::erase_snapshot(node);
    }

    // 8. Update statistics
    g_deletion_stats.nodes_deleted++;
    g_deletion_stats.memory_freed += mem_freed;

    // std::cout << "[careful_delete] Freed node@" << node
    //           << " op=" << op_name(node->op)
    //           << " mem=" << (mem_freed / 1024.0) << "KB"
    //           << " policy=" << (policy == DeletePolicy::AlwaysSafe ? "Safe" : "Aggressive")
    //           << "\n";
    return true;
}

// ============================================================================
// Smart Memory Pressure Detection
// ============================================================================

enum class MemoryPressure {
    LOW,     // < 50% of snapshots used
    MEDIUM,  // 50-80% of snapshots used
    HIGH     // > 80% of snapshots used
};

static MemoryPressure detect_memory_pressure() {
    size_t snapshot_mem = inplace::get_snapshot_memory_usage();
    // Simple heuristic: if snapshot memory is high, pressure is high
    // In production, this would check against available system memory
    
    if (snapshot_mem > 1024 * 1024 * 1024) {  // > 1GB
        return MemoryPressure::HIGH;
    } else if (snapshot_mem > 512 * 1024 * 1024) {  // > 512MB
        return MemoryPressure::MEDIUM;
    }
    return MemoryPressure::LOW;
}

// ============================================================================
// Sweep with Memory Pressure Awareness
// ============================================================================

void sweep_safe_nodes(const Value& root, DeletePolicy policy, const std::unordered_set<Node*>& protected_nodes) {
    if (!root.node) return;
    
    auto order = topo_from(root.node.get());
    int freed = 0;
    
    // First, try to cleanup stale snapshots
    size_t snapshot_freed = inplace::cleanup_stale_snapshots();
    if (snapshot_freed > 0) {
        std::cout << "[careful_delete] Cleaned up " << (snapshot_freed / 1024.0 / 1024.0) 
                  << " MB of stale snapshots\n";
    }
    
    // Detect memory pressure
    MemoryPressure pressure = detect_memory_pressure();
    
    // Adjust policy based on memory pressure
    DeletePolicy effective_policy = policy;
    if (pressure == MemoryPressure::HIGH && policy == DeletePolicy::AlwaysSafe) {
        std::cout << "[careful_delete] High memory pressure detected, using aggressive policy\n";
        effective_policy = DeletePolicy::Aggressive;
    }
    
    // Delete nodes
    for (Node* n : order) {
        // FIX: Always protect the root node (the result we are holding)
        if (n == root.node.get()) {
            continue;
        }
        
        if (try_delete_node(n, effective_policy, &protected_nodes))
            ++freed;
    }
    
    std::cout << "[careful_delete] Sweep complete. Freed " << freed << " nodes.\n";
}

// ============================================================================
// Smart Sweep with Checkpoint Priority
// ============================================================================

void sweep_with_checkpoint_priority(const Value& root, size_t target_memory_mb) {
    if (!root.node) return;
    
    auto order = topo_from(root.node.get());
    
    // Build list of non-checkpoint nodes sorted by memory usage
    struct NodeMemInfo {
        Node* node;
        size_t memory;
        bool is_checkpoint;
    };
    
    std::vector<NodeMemInfo> candidates;
    for (Node* n : order) {
        if (n->op == Op::Leaf) continue;
        if (has_active_alias(n)) continue;
        if (!gradients_done(n)) continue;
        
        NodeMemInfo info;
        info.node = n;
        info.memory = estimate_node_memory(n);
        info.is_checkpoint = n->is_checkpoint;
        candidates.push_back(info);
    }
    
    // Sort: non-checkpoints first, then by memory (largest first)
    std::sort(candidates.begin(), candidates.end(), 
              [](const NodeMemInfo& a, const NodeMemInfo& b) {
                  if (a.is_checkpoint != b.is_checkpoint)
                      return !a.is_checkpoint;  // non-checkpoints first
                  return a.memory > b.memory;   // larger memory first
              });
    
    // Delete until target reached
    size_t freed_mb = 0;
    size_t target_bytes = target_memory_mb * 1024 * 1024;
    
    for (const auto& info : candidates) {
        if (freed_mb >= target_bytes) break;
        
        DeletePolicy policy = info.is_checkpoint ? DeletePolicy::Aggressive : DeletePolicy::AlwaysSafe;
        if (try_delete_node(info.node, policy)) {
            freed_mb += info.memory;
        }
    }
    
    std::cout << "[careful_delete] Priority sweep freed " << (freed_mb / 1024.0 / 1024.0) << " MB\n";
}

// ============================================================================
// Debug and Statistics
// ============================================================================

void debug_deletion_state() {
    std::cout << "=== Careful Deletion Debug ===\n";
    g_deletion_stats.print();
    std::cout << "Snapshot memory: " << (inplace::get_snapshot_memory_usage() / 1024.0 / 1024.0) << " MB\n";
    
    MemoryPressure pressure = detect_memory_pressure();
    std::cout << "Memory pressure: ";
    switch (pressure) {
        case MemoryPressure::LOW: std::cout << "LOW\n"; break;
        case MemoryPressure::MEDIUM: std::cout << "MEDIUM\n"; break;
        case MemoryPressure::HIGH: std::cout << "HIGH\n"; break;
    }
    std::cout << "==============================\n";
}

void reset_deletion_stats() {
    g_deletion_stats.reset();
}

} // namespace memory
} // namespace ag