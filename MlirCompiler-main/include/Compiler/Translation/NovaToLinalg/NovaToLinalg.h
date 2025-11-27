#ifndef MLIR_CONVERSION_NOVATOLINALG_H
#define MLIR_CONVERSION_NOVATOLINALG_H

#include "mlir/Pass/Pass.h"
#include <memory>
namespace mlir{
    class Pass;
    class RewritePatternSet;
    class TypeConverter;
    namespace nova{
        std::unique_ptr<Pass> createNovaToLinalgLoweringPass();
        void regsiterNovaToLinalgLoweringTemplatePass();
        void populateNovaToLinalgPatterns(RewritePatternSet &patterns);
        void populateNovaToLinalgPatternsTemplate(RewritePatternSet &patterns);

    }
}
#endif
//Patterns
/*
gerenalized 
nova.add -> arith.addf/arith.addi
nova.sub -> arith.subf/arith.subi
nova.mul -> arith.mulf/arith.muli
nova.pow -> math.powf/math.ipowi
nova.sin -> math.sin
nova.broadcast_in_dim -> linalg generalized 
nova.matmul -> linalg.matmul

*/