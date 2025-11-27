#ifndef COMPILER_TRANSFORMS_FASTMATH_FLAG_H
#define COMPILER_TRANSFORMS_FASTMATH_FLAG_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace nova {

std::unique_ptr<Pass> createFastmathFlagPass();

} // namespace nova
} // namespace mlir

#endif // COMPILER_TRANSFORMS_FASTMATH_FLAG_H