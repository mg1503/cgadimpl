//============================================================
// file: cgadimpl/src/core/inplace.cpp
//============================================================
//
// This file implements the core logic for **in-place checkpointing,
// tensor versioning, and alias tracking** within the autodiff engine.
//
// It maintains consistency when tensors are modified in-place
// (e.g., via add_, mul_, or in-place parameter updates) while still
// supporting checkpoint recomputation, gradient propagation,
// and memory safety.
//
// The key mechanisms provided here are:
//   Version tracking — Detects when tensors have been modified.
//   Snapshot storage — Saves and restores safe copies of tensors.
//   Alias tracking — Handles tensors that share underlying memory.
//   Integration with checkpoint recomputation and careful deletion.
//
// It mirrors behavior similar to PyTorch’s internal `version counter`
// and `view tracking` systems.
//

#include "ad/inplace.hpp"
#include "ad/checkpoint.hpp"
#include "ad/debug.hpp"
#include <iostream>
#include <mutex>

namespace ag {
namespace inplace {

using NodePtr = std::shared_ptr<Node>; // alias for shared Node references

// -----------------------------------------------------------------------------
// Global Tracking Tables
// -----------------------------------------------------------------------------

/*
 * These global data structures maintain state about in-place operations.
 * They are shared across all nodes and guarded by a global mutex (g_lock)
 * to ensure thread safety.
 *
 * - g_snapshots : maps Node* → snapshot entry (Tensor + version at save)
 * - g_meta      : maps Node* → TensorMeta (current version + alias roots)
 * - g_alias     : maps storage pointer → set of all Nodes sharing that data
 */
static std::unordered_map<Node*, SnapshotEntry> g_snapshots;         
static std::unordered_map<Node*, TensorMeta>    g_meta;              
static std::unordered_map<void*, std::unordered_set<Node*>> g_alias; 
static std::mutex g_lock; // protects all global maps from concurrent access

// -----------------------------------------------------------------------------
// Internal Helper Functions
// -----------------------------------------------------------------------------

/*
 * register_tensor_alias():
 * -------------------------
 *  Registers that a given tensor storage (identified by its raw data pointer)
 *  belongs to a particular node. If multiple nodes share the same pointer,
 *  they are linked as *aliases* — meaning in-place modification to one affects all.
 *
 *  This forms a many-to-one mapping:
 *      storage_ptr → {nodes sharing this buffer}
 */
void register_tensor_alias(void* data_ptr, Node* node) {
    if (!node || !data_ptr) return;
    std::lock_guard<std::mutex> guard(g_lock);
    g_alias[data_ptr].insert(node);
    g_meta[node].alias_roots.insert(data_ptr);
}

/*
 * bump_tensor_version():
 * -----------------------
 *  Increments a node’s tensor version counter.
 *  Called after every in-place modification to detect stale snapshots later.
 */
void bump_tensor_version(Node* node) {
    if (!node) return;
    g_meta[node].version++;
}

/*
 * get_tensor_version():
 * ----------------------
 *  Retrieves the current version number of a node’s tensor.
 *  Returns 0 if no version metadata exists (new tensor or not tracked).
 */
size_t get_tensor_version(Node* node) {
    if (!node) return 0;
    auto it = g_meta.find(node);
    return (it == g_meta.end()) ? 0 : it->second.version;
}

/*
 * propagate_to_aliases():
 * ------------------------
 *  Synchronizes a recomputed tensor’s value with all other nodes
 *  that share the same data storage pointer (aliases).
 *
 *  For example, if node A and node B share memory, and A is recomputed,
 *  this ensures B’s value pointer is updated as well.
 */
static void propagate_to_aliases(Node* node, const Tensor& new_value) {
    if (!node) return;
    void* data_ptr = (void*)new_value.data();
    std::lock_guard<std::mutex> guard(g_lock);
    auto it = g_alias.find(data_ptr);
    if (it != g_alias.end()) {
        for (Node* alias_node : it->second) {
            alias_node->value = new_value; // shallow copy (shared buffer)
            g_meta[alias_node].version = g_meta[node].version;
        }
    }
}

// -----------------------------------------------------------------------------
// In-place Checkpoint Core
// -----------------------------------------------------------------------------

/*
 * mark_inplace_checkpoint():
 * ---------------------------
 *  Marks a node as an in-place checkpoint and stores a snapshot
 *  of its current tensor along with its version number.
 *
 *  This allows restoring the tensor later if it’s modified or deleted.
 *
 *  Steps:
 *    1 Mark node as checkpoint.
 *    2 Save minimal references to input nodes (for recomputation).
 *    3 Save a copy of its tensor as a snapshot (with version number).
 *    4 Register alias tracking for its data pointer.
 *    5 Optionally print debug info if `opts.verbose` is true.
 */
void mark_inplace_checkpoint(const NodePtr& node, const InplaceOptions& opts) {
    if (!node) return;
    if (node->is_checkpoint) return; // avoid re-marking

    node->is_checkpoint = true;

    // Save direct input references (like normal checkpoints)
    node->saved_inputs.clear();
    for (auto& p : node->inputs)
        node->saved_inputs.emplace_back(p ? Value(p) : Value());

    // Create snapshot entry
    SnapshotEntry entry;
    entry.snapshot = node->value;                    // store tensor copy
    entry.version_at_save = get_tensor_version(node.get());
    g_snapshots[node.get()] = entry;

    // Link alias tracking
    register_tensor_alias((void*)node->value.data(), node.get());

    if (opts.verbose) {
        std::cout << "[inplace] checkpoint @" << node.get()
                  << " shape=" << node->value.rows() << "x" << node->value.cols()
                  << " version=" << entry.version_at_save << "\n";
    }
}

// -----------------------------------------------------------------------------
// In-place Recompute Logic
// -----------------------------------------------------------------------------

/*
 * g_recompute_in_progress:
 * -------------------------
 *  Thread-local set of nodes currently being recomputed.
 *  Prevents recursive recomputation (which could cause infinite loops).
 */
static thread_local std::unordered_set<Node*> g_recompute_in_progress;

/*
 * recompute_inplace():
 * ---------------------
 *  Recomputes a node’s tensor if its stored snapshot is stale or missing.
 *  Integrates with `checkpoint_impl::recompute_subgraph()` to rebuild
 *  the tensor using its input dependencies.
 *
 *  Steps:
 *    1️⃣ Prevent recursive recompute (guard via thread-local set).
 *    2️⃣ Recompute forward pass of node via checkpoint subsystem.
 *    3️⃣ Update version and snapshot upon success.
 *    4️⃣ Propagate result to alias nodes.
 *    5️⃣ Clear in-progress flag.
 */
bool recompute_inplace(const NodePtr& node) {
    if (!node) return false;
    if (!node->is_checkpoint) return false;

    Node* raw = node.get();

    // Prevent infinite recursion (cycle guard)
    if (g_recompute_in_progress.find(raw) != g_recompute_in_progress.end()) {
        std::cerr << "[inplace][ERROR] recursive recompute detected for node@" << raw << "\n";
        return false;
    }

    g_recompute_in_progress.insert(raw);

    // Use checkpoint system to recompute without locking
    bool ok = ag::checkpoint_impl::recompute_subgraph(node);

    if (!ok) {
        std::cerr << "[inplace] recompute failed for node@" << raw << "\n";
        g_recompute_in_progress.erase(raw);
        return false;
    }

    // Update version and snapshot atomically
    {
        std::lock_guard<std::mutex> guard(g_lock);
        size_t current_ver = g_meta[raw].version;
        size_t new_ver = current_ver + 1;
        g_meta[raw].version = new_ver;
        g_snapshots[raw] = { node->value, new_ver };
    }

    // Update any aliased nodes
    propagate_to_aliases(raw, node->value);

    // Done
    g_recompute_in_progress.erase(raw);
    return true;
}

// -----------------------------------------------------------------------------
// ensure_inplace_value()
// -----------------------------------------------------------------------------

/*
 * ensure_inplace_value():
 * ------------------------
 *  Ensures that a node’s tensor value exists and is valid.
 *  It checks snapshots, compares versions, and if necessary,
 *  triggers recomputation or restoration.
 *
 *  Flow:
 *    - If tensor is already valid → return immediately.
 *    - Else, if a snapshot exists → restore if up-to-date, or recompute if stale.
 *    - Else, if no snapshot but node is checkpointed → recompute.
 *    - Else → nothing can be done (tensor unavailable).
 */
bool ensure_inplace_value(const NodePtr& node) {
    if (!node) return false;
    Node* raw = node.get();

    std::cerr << "[inplace] ensure enter node@" << raw << "\n";

    // Fast path: tensor already present
    if (node->value.size() != 0) {
        std::cerr << "[inplace] node@" << raw << " already has value\n";
        return true;
    }

    // Retrieve snapshot safely (without holding lock during recompute)
    SnapshotEntry snap;
    {
        std::lock_guard<std::mutex> guard(g_lock);
        auto sit = g_snapshots.find(raw);
        if (sit != g_snapshots.end()) snap = sit->second;
    }
    bool has_snapshot = (snap.snapshot.size() != 0);

    size_t meta_ver = get_tensor_version(raw);

    if (has_snapshot) {
        std::cerr << "[inplace] found snapshot for node@" << raw
                  << " snap_ver=" << snap.version_at_save
                  << " meta_ver=" << meta_ver << "\n";

        // Case 1: snapshot is valid
        if (meta_ver == 0 || snap.version_at_save == meta_ver) {
            node->value = snap.snapshot;
            propagate_to_aliases(raw, node->value);
            std::cerr << "[inplace] restored snapshot for node@" << raw << "\n";
            {
                std::lock_guard<std::mutex> guard(g_lock);
                if (g_meta.find(raw) == g_meta.end())
                    g_meta[raw].version = snap.version_at_save;
            }
            return true;
        }

        // Case 2: snapshot is outdated → prefer recomputation
        std::cerr << "[inplace] snapshot stale (snap=" << snap.version_at_save
                  << " meta=" << meta_ver << ") -> recompute\n";
        bool recomputed = recompute_inplace(node);
        if (recomputed) {
            std::cerr << "[inplace] recompute succeeded for node@" << raw << "\n";
            return true;
        }

        // Case 3: recompute failed → fallback to snapshot anyway
        std::cerr << "[inplace] recompute failed; fallback to snapshot\n";
        node->value = snap.snapshot;
        propagate_to_aliases(raw, node->value);
        return true;
    }

    // Case 4: no snapshot but checkpointed → recompute
    if (node->is_checkpoint) {
        std::cerr << "[inplace] no snapshot, attempting recompute\n";
        bool rc = recompute_inplace(node);
        if (!rc) std::cerr << "[inplace] recompute failed (no snapshot)\n";
        return rc;
    }

    // Case 5: no snapshot and not checkpointed
    std::cerr << "[inplace] no snapshot and not checkpointed for node@" << raw << "\n";
    return false;
}

// -----------------------------------------------------------------------------
// on_recomputed(): public hook called after recomputation
// -----------------------------------------------------------------------------

/*
 * on_recomputed():
 * -----------------
 *  Updates global metadata when a node is recomputed externally
 *  (for example, by `checkpoint.cpp`).
 *
 *  It increments the version, updates the snapshot,
 *  and propagates the new tensor to aliases.
 */
void on_recomputed(Node* raw) {
    if (!raw) return;
    {
        std::lock_guard<std::mutex> guard(g_lock);
        size_t new_ver = g_meta[raw].version + 1;
        g_meta[raw].version = new_ver;
        g_snapshots[raw] = { raw->value, new_ver };
    }
    propagate_to_aliases(raw, raw->value);
}

/*
 * clear_inplace_checkpoints():
 * -----------------------------
 *  Clears all internal tables — used when resetting or rebuilding the graph.
 *  Frees all snapshots, metadata, and alias records.
 */
void clear_inplace_checkpoints() {
    std::lock_guard<std::mutex> guard(g_lock);
    g_snapshots.clear();
    g_meta.clear();
    g_alias.clear();
}

// -----------------------------------------------------------------------------
// Debug Utilities
// -----------------------------------------------------------------------------

/*
 * debug_alias_table():
 * ---------------------
 *  Prints a human-readable summary of all alias groups and their nodes.
 *  Shows which storage addresses are shared and each node’s version.
 */
void debug_alias_table() {
    std::lock_guard<std::mutex> guard(g_lock);
    std::cout << "=== Inplace alias table ===\n";
    for (auto& [ptr, nodes] : g_alias) {
        std::cout << "Storage@" << ptr << " shared by " << nodes.size() << " nodes\n";
        for (Node* n : nodes)
            std::cout << "   node@" << n << " version=" << get_tensor_version(n) << "\n";
    }
}

// -----------------------------------------------------------------------------
// detail namespace — helpers for memory management
// -----------------------------------------------------------------------------
namespace detail {

/*
 * has_alias():
 * -------------
 *  Checks whether a node is part of any alias group.
 *  Used by the memory manager (careful_deletion.cpp)
 *  to decide if freeing a node’s tensor is safe.
 */
bool has_alias(Node* node) {
    if (!node) return false;
    std::lock_guard<std::mutex> guard(g_lock);
    for (auto& [ptr, nodes] : g_alias) {
        if (nodes.find(node) != nodes.end())
            return true;
    }
    return false;
}

/*
 * erase_snapshot():
 * ------------------
 *  Deletes a node’s snapshot entry from g_snapshots if it exists.
 *  Used by aggressive deletion policies to reclaim memory quickly.
 */
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

// -----------------------------------------------------------------------------
// debug namespace — runtime visualization tools
// -----------------------------------------------------------------------------
namespace debug {

/*
 * print_version_table():
 * -----------------------
 *  Prints the complete version tracking table.
 *  Displays each node’s pointer, operation type, and current version.
 *
 *  Useful for verifying that version counters increment correctly
 *  after in-place operations or recomputation events.
 */
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
