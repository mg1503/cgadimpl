#pragma once
#include <string>
#include "ad/graph.hpp"

namespace ag::hlo {

// Dump the compute graph rooted at `root` as StableHLO-like MLIR.
// All Leaf nodes (requires_grad==true or false) are exported as function arguments.
// The result is the tensor value at `root`.
void dump_stablehlo(const Value& root, const std::string& filepath);

} // namespace ag::hlo
