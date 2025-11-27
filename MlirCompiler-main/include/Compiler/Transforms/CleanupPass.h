#ifndef COMPILER_TRANSFORMS_CLEANUP_PASS_H
#define COMPILER_TRANSFORMS_CLEANUP_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir::compiler {

std::unique_ptr<Pass> createCleanupPass();

} // namespace mlir::compiler

#endif // COMPILER_TRANSFORMS_CLEANUP_PASS_H