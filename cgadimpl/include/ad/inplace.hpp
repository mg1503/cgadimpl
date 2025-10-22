//============================================================
// file: cgadimpl/include/ad/inplace.hpp
//============================================================
#pragma once
#include "ad/graph.hpp"
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <vector>
#include <iostream>

namespace ag {
namespace inplace {

/*
 * ============================================================
 * Purpose and Overview
 * ============================================================
 * 
 * This header defines the **in-place checkpointing, versioning, and alias-tracking system**
 * for the autodiff engine.
 *
 * In-place operations (like `add_`, `mul_`, etc.) modify tensor data directly.
 * While this improves performance and saves memory, it also introduces risks:
 *    - It can invalidate saved activations used for gradient computation.
 *    - It can break the correctness of checkpoint recomputation.
 *    - It can make alias detection (shared data between tensors) tricky.
 *
 * To handle these issues, this subsystem provides:
 *    Version tracking: assigns a version number to every tensor and bumps it
 *      whenever an in-place modification occurs.
 *
 *    Alias tracking: records which tensors share the same memory storage
 *      (important for detecting when a change affects multiple nodes).
 *
 *    Snapshot checkpointing: stores copies ("snapshots") of tensors
 *      to allow safe recomputation after in-place modifications or deletions.
 *
 * Together, these features make the system behave similarly to
 * PyTorch’s “View + Version Counter” and TensorFlow’s alias tracking mechanism.
 *
 * This file only defines the API and structures; their implementations are
 * provided in `inplace.cpp`.
 */

// -----------------------------------------------------------------------------
// Options for In-Place Checkpointing
// -----------------------------------------------------------------------------

/*
 *  InplaceOptions:
 *  ----------------
 *  Configures how in-place checkpoints behave. These options control
 *  whether deltas, RNG states, or verbose logs are stored during
 *  checkpoint operations.
 *
 *  Fields:
 *    - store_delta : If true, the system stores only the *difference (delta)*
 *                    between the saved tensor and the current tensor.
 *                    This can reduce memory usage when differences are small.
 *
 *    - save_rng    : If true, the system also stores the RNG state (similar
 *                    to checkpointing) so that recomputation remains deterministic
 *                    even if random operations occurred between checkpoints.
 *
 *    - verbose     : If true, prints debug information during
 *                    checkpoint marking, recomputation, and alias tracking.
 */
struct InplaceOptions {
    bool store_delta = true;   // Whether to store differential deltas instead of full tensors.
    bool save_rng    = false;  // Whether to save RNG state for deterministic replay.
    bool verbose     = false;  // Whether to print debug messages for developers.
};

// -----------------------------------------------------------------------------
// Versioning and Alias Tracking Structures
// -----------------------------------------------------------------------------

/*
 *  TensorMeta:
 *  ------------
 *  Tracks versioning and alias relationships for tensors associated with nodes.
 *
 *  Each node’s tensor has a corresponding version number that increments
 *  whenever an in-place modification occurs (like `add_`, `mul_`, etc.).
 *
 *  Fields:
 *    - version     : A monotonically increasing counter indicating how many
 *                    times this tensor has been modified in place.
 *
 *    - alias_roots : A set of memory addresses (pointers to data buffers)
 *                    that share the same underlying storage.
 *                    If two tensors share a `data_ptr`, they belong to the same alias group.
 *
 *  Purpose:
 *    - Enables automatic detection of unsafe aliasing.
 *    - Allows recomputation or warning when tensors change unexpectedly.
 */
struct TensorMeta {
    size_t version = 0;                      // Incremented after every in-place modification.
    std::unordered_set<void*> alias_roots;   // All other tensors that share storage with this tensor.
};

/*
 *  SnapshotEntry:
 *  ----------------
 *  Represents a saved tensor state and its corresponding version number
 *  at the moment it was checkpointed.
 *
 *  Fields:
 *    - snapshot        : The actual Tensor copy or snapshot of the data.
 *    - version_at_save : The version number when the snapshot was taken.
 *
 *  Purpose:
 *    - If the tensor’s version changes later (due to in-place ops),
 *      the system can detect that the saved snapshot is outdated
 *      and trigger recomputation.
 */
struct SnapshotEntry {
    Tensor snapshot;          // Saved tensor data (copy or delta-based).
    size_t version_at_save = 0;  // Version when snapshot was created.
};

// -----------------------------------------------------------------------------
// Inplace Checkpoint API
// -----------------------------------------------------------------------------

/*
 *  on_recomputed():
 *  -----------------
 *  Called whenever a node’s value has been recomputed (for example,
 *  after a checkpoint restoration or lazy evaluation).
 *
 *  This updates internal tables so that future in-place operations
 *  know the tensor has been refreshed and is now safe to modify.
 */
void on_recomputed(Node* node);

/*
 *  mark_inplace_checkpoint():
 *  ---------------------------
 *  Marks a node as an in-place checkpoint, saving its tensor snapshot
 *  and version metadata according to the provided options.
 *
 *  Parameters:
 *      - node : shared_ptr<Node> → target node
 *      - opts : InplaceOptions   → configuration options
 *
 *  Purpose:
 *      - Creates a "safe restore point" for the node’s tensor.
 *      - Ensures that if the tensor is modified later, we can restore it
 *        or recompute it accurately using the saved snapshot.
 */
void mark_inplace_checkpoint(const std::shared_ptr<Node>& node,
                             const InplaceOptions& opts = {});

/*
 *  ensure_inplace_value():
 *  ------------------------
 *  Ensures that a node’s tensor value is present and valid.
 *  If the tensor has been deleted or its version is outdated,
 *  this function attempts to restore it from the saved snapshot
 *  or recompute it using dependency information.
 *
 *  Returns:
 *      true  → if the value is present and up-to-date.
 *      false → if restoration failed (e.g., snapshot missing).
 */
bool ensure_inplace_value(const std::shared_ptr<Node>& node);

/*
 *  recompute_inplace():
 *  ---------------------
 *  Explicitly triggers recomputation of a node’s tensor
 *  if its snapshot is stale or missing.
 *  After recomputation, the snapshot and version counter are refreshed.
 *
 *  Returns:
 *      true  → successful recomputation.
 *      false → recomputation failed or data inconsistent.
 */
bool recompute_inplace(const std::shared_ptr<Node>& node);

/*
 *  clear_inplace_checkpoints():
 *  -----------------------------
 *  Clears all saved in-place checkpoints, snapshots, and alias tracking
 *  metadata from the system.
 *
 *  Use this when resetting the graph or cleaning up between training iterations.
 */
void clear_inplace_checkpoints();

// -----------------------------------------------------------------------------
// Versioning and Alias Tracking API
// -----------------------------------------------------------------------------

/*
 *  register_tensor_alias():
 *  -------------------------
 *  Registers a tensor’s memory storage (`data_ptr`) and associates it
 *  with the node that owns it. If another tensor uses the same pointer,
 *  both are linked in the alias table.
 *
 *  Purpose:
 *      - Detects alias groups (tensors sharing memory).
 *      - Prevents unsafe deletions or modifications of shared tensors.
 *
 *  Parameters:
 *      - data_ptr : raw pointer to tensor’s data buffer.
 *      - node     : Node associated with that storage.
 */
void register_tensor_alias(void* data_ptr, Node* node);

/*
 *  bump_tensor_version():
 *  -----------------------
 *  Increments the tensor’s version number after an in-place operation.
 *  This is used to detect whether a tensor has been modified
 *  since it was last checkpointed.
 */
void bump_tensor_version(Node* node);

/*
 *  get_tensor_version():
 *  ----------------------
 *  Retrieves the current version number of the tensor associated with the node.
 *  Returns 0 if the node is not registered in the version table.
 */
size_t get_tensor_version(Node* node);

/*
 *  debug_alias_table():
 *  ---------------------
 *  Prints the current alias and version tracking table.
 *  Helpful for debugging complex aliasing or memory-sharing issues.
 */
void debug_alias_table();

// -----------------------------------------------------------------------------
// Public Helper Accessors for Memory Management Modules
// -----------------------------------------------------------------------------
namespace detail {

/*
 *  has_alias():
 *  -------------
 *  Checks whether the given node is part of any alias group.
 *  Used by modules like `careful_deletion.cpp` to decide
 *  whether deleting a tensor is safe.
 *
 *  Returns:
 *      true  → if the node’s tensor shares storage with others.
 *      false → otherwise.
 */
bool has_alias(Node* node);

/*
 *  erase_snapshot():
 *  ------------------
 *  Deletes (erases) any saved snapshot entry for the given node,
 *  freeing up its stored checkpoint data.
 *
 *  Used in aggressive cleanup policies (e.g., memory::DeletePolicy::Aggressive)
 *  to reclaim storage associated with obsolete checkpoints.
 *
 *  Returns:
 *      true  → if a snapshot was found and erased.
 *      false → if no snapshot existed for the node.
 */
bool erase_snapshot(Node* node);

} // namespace detail

// -----------------------------------------------------------------------------
// Debugging Utilities
// -----------------------------------------------------------------------------
namespace debug {

/*
 *  print_version_table():
 *  -----------------------
 *  Dumps the current tensor version tracking table for human inspection.
 *  This includes node identifiers, their version numbers,
 *  and any alias groups they belong to.
 *
 *  Useful for verifying version consistency after in-place operations.
 */
void print_version_table();

} // namespace debug

} // namespace inplace
} // namespace ag
