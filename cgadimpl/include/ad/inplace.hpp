#pragma once
#include "ad/graph.hpp"
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <vector>
#include <iostream>

namespace ag {
namespace inplace {

// -----------------------------------------------------------------------------
// Options for in-place checkpointing
// -----------------------------------------------------------------------------
struct InplaceOptions {
    bool store_delta = true;   // store delta between saved and current tensor
    bool save_rng    = false;  // optionally save RNG state
    bool verbose     = false;  // print debug info
};

// -----------------------------------------------------------------------------
// Versioning and Alias tracking types
// -----------------------------------------------------------------------------

// Each tensor stored in a node gets a version number that increments whenever
// it's modified in-place (e.g., add_, mul_, etc.)
struct TensorMeta {
    size_t version = 0;                      // incremented on each in-place op
    std::unordered_set<void*> alias_roots;   // track all other tensors sharing same data
};

// Snapshot entry: stores tensor state and version number at checkpoint
struct SnapshotEntry {
    Tensor snapshot;
    size_t version_at_save = 0;
};

// -----------------------------------------------------------------------------
// Inplace checkpoint API
// -----------------------------------------------------------------------------
void on_recomputed(Node* node);
// Mark a node for in-place checkpointing
void mark_inplace_checkpoint(const std::shared_ptr<Node>& node,
                             const InplaceOptions& opts = {});

// Restore or recompute value for a node whose tensor was freed/evicted
bool ensure_inplace_value(const std::shared_ptr<Node>& node);

// Recompute in-place and refresh stored snapshot
bool recompute_inplace(const std::shared_ptr<Node>& node);

// Clear all stored inplace snapshots and alias tracking
void clear_inplace_checkpoints();

// -----------------------------------------------------------------------------
// Versioning + Alias tracking API
// -----------------------------------------------------------------------------

// Register a tensor's storage address and link all aliases
void register_tensor_alias(void* data_ptr, Node* node);

// Increment tensor version (called after any in-place modification)
void bump_tensor_version(Node* node);

// Get the current tensor version for a node
size_t get_tensor_version(Node* node);

// Print diagnostic info (optional)
void debug_alias_table();

// -----------------------------------------------------------------------------
// Public helper accessors for memory management modules
// -----------------------------------------------------------------------------
namespace detail {
// Returns true if given node is part of any alias group.
bool has_alias(Node* node);

// Deletes snapshot entry for node if exists.
bool erase_snapshot(Node* node);
}
namespace debug {
// Returns a snapshot of the current version table for inspection.
void print_version_table();
}


} // namespace inplace
} // namespace ag
