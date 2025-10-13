#include "ad/inplace.hpp"
#include "ad/checkpoint.hpp"
#include "ad/debug.hpp"
#include <iostream>
#include <mutex>

namespace ag {
namespace inplace {

using NodePtr = std::shared_ptr<Node>;

// -----------------------------------------------------------------------------
// Global tracking tables
// -----------------------------------------------------------------------------
static std::unordered_map<Node*, SnapshotEntry> g_snapshots;         // Node → snapshot + version
static std::unordered_map<Node*, TensorMeta>    g_meta;              // Node → version/alias info
static std::unordered_map<void*, std::unordered_set<Node*>> g_alias; // data_ptr → all nodes sharing it
static std::mutex g_lock;

// -----------------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------------

// Link aliases: if multiple tensors share same data_ptr, they are considered views.
void register_tensor_alias(void* data_ptr, Node* node) {
    if (!node || !data_ptr) return;
    std::lock_guard<std::mutex> guard(g_lock);
    g_alias[data_ptr].insert(node);
    g_meta[node].alias_roots.insert(data_ptr);
}

// Increment tensor version counter
void bump_tensor_version(Node* node) {
    if (!node) return;
    g_meta[node].version++;
}

// Get tensor version safely
size_t get_tensor_version(Node* node) {
    if (!node) return 0;
    auto it = g_meta.find(node);
    return (it == g_meta.end()) ? 0 : it->second.version;
}

// Propagate new snapshot to all aliases sharing same memory
static void propagate_to_aliases(Node* node, const Tensor& new_value) {
    if (!node) return;
    void* data_ptr = (void*)new_value.data();
    std::lock_guard<std::mutex> guard(g_lock);
    auto it = g_alias.find(data_ptr);
    if (it != g_alias.end()) {
        for (Node* alias_node : it->second) {
            alias_node->value = new_value; // shallow copy shares same storage
            g_meta[alias_node].version = g_meta[node].version;
        }
    }
}

// -----------------------------------------------------------------------------
// Checkpoint core functions
// -----------------------------------------------------------------------------

void mark_inplace_checkpoint(const NodePtr& node, const InplaceOptions& opts) {
    if (!node) return;
    if (node->is_checkpoint) return;
    node->is_checkpoint = true;

    // Save references to inputs (like normal checkpoint)
    node->saved_inputs.clear();
    for (auto& p : node->inputs)
        node->saved_inputs.emplace_back(p ? Value(p) : Value());

    // Store snapshot and version
    SnapshotEntry entry;
    entry.snapshot = node->value;
    entry.version_at_save = get_tensor_version(node.get());
    g_snapshots[node.get()] = entry;

    // Register alias tracking for this tensor
    register_tensor_alias((void*)node->value.data(), node.get());

    if (opts.verbose) {
        std::cout << "[inplace] checkpoint @" << node.get()
                  << " shape=" << node->value.rows() << "x" << node->value.cols()
                  << " version=" << entry.version_at_save << "\n";
    }
}
// --- recompute_inplace -----------------------------------------------------
// bool recompute_inplace(const NodePtr& node) {
//     if (!node) return false;
//     if (!node->is_checkpoint) return false;

//     // Run forward recomputation (uses checkpoint_impl::recompute_subgraph)
//     if (!ag::checkpoint_impl::recompute_subgraph(node)) {
//         std::cerr << "[inplace] recompute failed @" << node.get() << "\n";
//         return false;
//     }

//     // Always increment version when we recompute (one increment per recompute)
//     {
//         std::lock_guard<std::mutex> guard(g_lock);
//         size_t current_ver = 0;
//         auto mit = g_meta.find(node.get());
//         if (mit != g_meta.end()) current_ver = mit->second.version;
//         size_t new_ver = current_ver + 1;
//         g_meta[node.get()].version = new_ver;

//         // Update snapshot to the recomputed value and tag with new_ver
//         g_snapshots[node.get()] = { node->value, new_ver };
//     }

//     // Propagate the recomputed value to aliases (shares storage pointer)
//     propagate_to_aliases(node.get(), node->value);
//     return true;
// }

// // --- ensure_inplace_value --------------------------------------------------
// bool ensure_inplace_value(const NodePtr& node) {
//     if (!node) return false;

//     // 1) If node->value is already present, nothing to do.
//     if (node->value.size() != 0) return true;

//     // 2) If we have a saved snapshot, inspect it
//     SnapshotEntry snap;
//     {
//         std::lock_guard<std::mutex> guard(g_lock);
//         auto sit = g_snapshots.find(node.get());
//         if (sit != g_snapshots.end()) snap = sit->second;
//         else snap = SnapshotEntry(); // no snapshot
//     }

//     bool has_snapshot = (snap.snapshot.size() != 0);

//     // If we have a snapshot, compare versions
//     if (has_snapshot) {
//         size_t meta_ver = get_tensor_version(node.get()); // current known meta version (0 if not present)
//         if (meta_ver == 0) {
//             // No prior meta recorded — safe to restore snapshot
//             node->value = snap.snapshot;
//             // set meta version to saved snapshot version if not present
//             std::lock_guard<std::mutex> guard(g_lock);
//             if (g_meta.find(node.get()) == g_meta.end()) {
//                 g_meta[node.get()].version = snap.version_at_save;
//             }
//             propagate_to_aliases(node.get(), node->value);
//             return true;
//         } else {
//             // meta_ver > 0 means node has a recorded version.
//             // If snapshot was saved at the SAME version, it's safe to restore.
//             // If snapshot is older (version_at_save < meta_ver), prefer recompute.
//             if (snap.version_at_save == meta_ver) {
//                 node->value = snap.snapshot;
//                 propagate_to_aliases(node.get(), node->value);
//                 return true;
//             } else {
//                 // snapshot is stale (older than current meta) — recompute instead.
//                 if (recompute_inplace(node)) return true;
//                 // If recompute fails, as a fallback try restoring snapshot (but don't downgrade meta).
//                 node->value = snap.snapshot;
//                 propagate_to_aliases(node.get(), node->value);
//                 return true;
//             }
//         }
//     }

//     // 3) If no snapshot, fall back to recompute (if node marked checkpoint)
//     if (node->is_checkpoint) {
//         return recompute_inplace(node);
//     }

//     // No snapshot and not checkpointed — cannot restore
//     return false;
// }
// at top of inplace.cpp (near other statics), add:
static thread_local std::unordered_set<Node*> g_recompute_in_progress;

// --- recompute_inplace -----------------------------------------------------
bool recompute_inplace(const NodePtr& node) {
    if (!node) return false;
    if (!node->is_checkpoint) return false;

    Node* raw = node.get();

    // Re-entry guard: prevent infinite recursion
    if (g_recompute_in_progress.find(raw) != g_recompute_in_progress.end()) {
        std::cerr << "[inplace][ERROR] recursive recompute detected for node@" << raw << "\n";
        return false; // prevent recursion; caller can fallback to snapshot restore
    }

    // Mark in-progress
    g_recompute_in_progress.insert(raw);

    // Perform recompute WITHOUT holding global lock (avoid deadlocks)
    bool ok = ag::checkpoint_impl::recompute_subgraph(node);

    if (!ok) {
        std::cerr << "[inplace] recompute failed for node@" << raw << "\n";
        g_recompute_in_progress.erase(raw);
        return false;
    }

    // Update version and snapshot under lock
    {
        std::lock_guard<std::mutex> guard(g_lock);
        size_t current_ver = 0;
        auto mit = g_meta.find(raw);
        if (mit != g_meta.end()) current_ver = mit->second.version;
        size_t new_ver = current_ver + 1;
        g_meta[raw].version = new_ver;
        g_snapshots[raw] = { node->value, new_ver };
    }

    // propagate to aliases (no global metadata lock required)
    propagate_to_aliases(raw, node->value);

    // Clear in-progress marker
    g_recompute_in_progress.erase(raw);
    return true;
}

// --- ensure_inplace_value --------------------------------------------------
bool ensure_inplace_value(const NodePtr& node) {
    if (!node) return false;
    Node* raw = node.get();

    std::cerr << "[inplace] ensure enter node@" << raw << "\n";

    // If node already has value, nothing to do
    if (node->value.size() != 0) {
        std::cerr << "[inplace] node@" << raw << " already has value\n";
        return true;
    }

    // Copy snapshot entry under lock (so we avoid holding lock across recompute)
    SnapshotEntry snap;
    {
        std::lock_guard<std::mutex> guard(g_lock);
        auto sit = g_snapshots.find(raw);
        if (sit != g_snapshots.end()) snap = sit->second;
    }
    bool has_snapshot = (snap.snapshot.size() != 0);

    // Current meta version (0 if none)
    size_t meta_ver = get_tensor_version(raw);

    if (has_snapshot) {
        std::cerr << "[inplace] found snapshot for node@" << raw
                  << " snap_ver=" << snap.version_at_save
                  << " meta_ver=" << meta_ver << "\n";

        // If snapshot matches current meta -> safe to restore
        if (meta_ver == 0 || snap.version_at_save == meta_ver) {
            node->value = snap.snapshot;
            propagate_to_aliases(raw, node->value);
            std::cerr << "[inplace] restored snapshot for node@" << raw << "\n";
            // If meta absent, set it to snapshot version
            {
                std::lock_guard<std::mutex> guard(g_lock);
                if (g_meta.find(raw) == g_meta.end()) g_meta[raw].version = snap.version_at_save;
            }
            return true;
        }

        // snapshot is older than current meta -> prefer recompute
        std::cerr << "[inplace] snapshot stale (snap=" << snap.version_at_save
                  << " meta=" << meta_ver << ") for node@" << raw << " -> recompute\n";

        // Attempt recompute, but avoid recursive recompute
        bool recomputed = recompute_inplace(node);
        if (recomputed) {
            std::cerr << "[inplace] recompute succeeded for node@" << raw << "\n";
            return true;
        }

        // recompute failed (or detected recursion) -> fallback to snapshot restore (but do not downgrade version)
        std::cerr << "[inplace] recompute failed or recursive; falling back to snapshot for node@" << raw << "\n";
        node->value = snap.snapshot;
        propagate_to_aliases(raw, node->value);
        return true;
    }

    // No snapshot: if node is a checkpoint, attempt recompute
    if (node->is_checkpoint) {
        std::cerr << "[inplace] no snapshot, attempting recompute for node@" << raw << "\n";
        bool rc = recompute_inplace(node);
        if (!rc) std::cerr << "[inplace] recompute failed (no snapshot) for node@" << raw << "\n";
        return rc;
    }

    // Nothing to do
    std::cerr << "[inplace] no snapshot and not checkpointed for node@" << raw << "\n";
    return false;
}
// public: called by other modules when recompute completed
void on_recomputed(Node* raw) {
    if (!raw) return;
    // Update meta/snapshot under lock
    {
        std::lock_guard<std::mutex> guard(g_lock);
        size_t current_ver = 0;
        auto mit = g_meta.find(raw);
        if (mit != g_meta.end()) current_ver = mit->second.version;
        size_t new_ver = current_ver + 1;
        g_meta[raw].version = new_ver;
        g_snapshots[raw] = { raw->value, new_ver };
    }
    // propagate value to aliases
    propagate_to_aliases(raw, raw->value);
}

void clear_inplace_checkpoints() {
    std::lock_guard<std::mutex> guard(g_lock);
    g_snapshots.clear();
    g_meta.clear();
    g_alias.clear();
}

// -----------------------------------------------------------------------------
// Debug utilities
// -----------------------------------------------------------------------------
void debug_alias_table() {
    std::lock_guard<std::mutex> guard(g_lock);
    std::cout << "=== Inplace alias table ===\n";
    for (auto& [ptr, nodes] : g_alias) {
        std::cout << "Storage@" << ptr << " shared by " << nodes.size() << " nodes\n";
        for (Node* n : nodes)
            std::cout << "   node@" << n << " version=" << get_tensor_version(n) << "\n";
    }
}

namespace detail {

bool has_alias(Node* node) {
    if (!node) return false;
    std::lock_guard<std::mutex> guard(g_lock);
    for (auto& [ptr, nodes] : g_alias) {
        if (nodes.find(node) != nodes.end())
            return true;
    }
    return false;
}

bool erase_snapshot(Node* node) {
    if (!node) return false;
    std::lock_guard<std::mutex> guard(g_lock);
    auto it = g_snapshots.find(node);
    if (it != g_snapshots.end()) {
        g_snapshots.erase(it);
        return true;
    }
    return false;
}

} // namespace detail
namespace debug {

void print_version_table() {
    std::lock_guard<std::mutex> guard(g_lock);
    std::cout << "\n=== Inplace Version Table Dump ===\n";
    for (auto& [node_ptr, meta] : g_meta) {
        std::cout << " Node@" << node_ptr
                  << " op=" << op_name(node_ptr->op)
                  << " version=" << meta.version << "\n";
    }
    std::cout << "===============================\n";
}

} // namespace debug

} // namespace inplace
} // namespace ag
