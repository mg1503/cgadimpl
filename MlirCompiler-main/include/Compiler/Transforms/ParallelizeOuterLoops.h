#ifndef COMPILER_TRANSFORMS_PARALLELIZE_OUTER_LOOPS_H_
#define COMPILER_TRANSFORMS_PARALLELIZE_OUTER_LOOPS_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace nova {

std::unique_ptr<Pass> createParallelizeOuterLoopsPass();

} // namespace nova
} // namespace mlir

#endif // COMPILER_TRANSFORMS_PARALLELIZE_OUTER_LOOPS_H_
