#ifndef NOVA_UTILS_BROADCAST_H
#define NOVA_UTILS_BROADCAST_H

#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <cstdint>

namespace mlir {
namespace nova {
/// Compute the broadcasted shape of two input shapes according to NumPy rules.
/// Returns std::nullopt if shapes are not broadcast-compatible.

std::optional<llvm::SmallVector<int64_t, 4>> 
computeBroadcastShape(llvm::ArrayRef<int64_t> shape1,
                      llvm::ArrayRef<int64_t> shape2);

bool isBroadcastCompatible(llvm::ArrayRef<int64_t> shape1,
                           llvm::ArrayRef<int64_t> shape2);

llvm::SmallVector<int64_t>
computeBroadcastDimensions(int64_t rank1, int64_t rank2);
}
}
#endif