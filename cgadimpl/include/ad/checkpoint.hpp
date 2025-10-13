// include/ad/checkpoint.hpp
#pragma once
#include <vector>
#include <memory>
#include <cstdint>
#include "ad/ops.hpp"
#include "ad/graph.hpp"
#include "ad/schema.hpp"

namespace ag {

struct CheckpointOptions {
    bool save_rng = true;         // save RNG state to ensure deterministic recompute (useful for dropout)
    int max_recompute_depth = 1000;
    bool save_inputs = true;
    bool detach_inputs = false;
    bool force = false;
};

// Value checkpoint(const Value &v, const CheckpointOptions &opts = CheckpointOptions());

void auto_checkpoint_every_n(const Value &root, int n);

void auto_checkpoint_by_depth(const Value& root, int depth_threshold);

// Internal implementation namespace (used by core code)
namespace checkpoint_impl {

void mark_node_checkpoint(const std::shared_ptr<Node> &node, const CheckpointOptions &opts = CheckpointOptions());

bool recompute_subgraph(const std::shared_ptr<Node>& node);

inline bool is_checkpointed(const std::shared_ptr<Node> &node);

} // namespace checkpoint_impl
} // namespace ag
