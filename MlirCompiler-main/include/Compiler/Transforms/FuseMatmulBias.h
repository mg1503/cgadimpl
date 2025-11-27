#ifndef COMPILER_TRANSFORMS_FUSE_MATMUL_BIAS_H_
#define COMPILER_TRANSFORMS_FUSE_MATMUL_BIAS_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace nova {

std::unique_ptr<Pass> createFuseMatmulBiasPass();

} // namespace nova
} // namespace mlir

#endif // COMPILER_TRANSFORMS_FUSE_MATMUL_BIAS_H_
