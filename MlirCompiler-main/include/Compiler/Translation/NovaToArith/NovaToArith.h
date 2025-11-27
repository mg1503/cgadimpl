
#ifndef MLIR_CONVERSION_NOVATOARITH_NOVATOARITH_H
#define MLIR_CONVERSION_NOVATOARITH_NOVATOARITH_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;
class TypeConverter;

namespace nova {

//===----------------------------------------------------------------------===//
// Pass Creation
//===-------------------------------------------------------------------------===//

/// Create a pass to lower Nova dialect operations to Arith dialect operations.
/// This pass converts:
///   - nova.add -> arith.addi/arith.addf
std::unique_ptr<Pass> createNovaToArithLoweringPass();

/// Register the Nova to Arith lowering pass.
void registerNovaToArithLoweringPass();


/// Populate the given pattern set with patterns that convert Nova ops to Arith ops.
/// This is useful if you want to integrate these patterns into a larger conversion pass.
void populateNovaToArithConversionPatterns(RewritePatternSet &patterns);



} // namespace nova
} // namespace mlir

#endif 