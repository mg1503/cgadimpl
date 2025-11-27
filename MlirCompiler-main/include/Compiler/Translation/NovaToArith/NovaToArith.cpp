//===- NovaToArithLowering.cpp - Lower Nova dialect to Arith dialect ------===//
//
// This file implements a pass to lower Nova dialect operations to Arith dialect
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"


#include "Compiler/Translation/NovaToArith/NovaToArith.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Dialect/nova/NovaDialect.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace nova {

//--------------------------------------constant-----------------------------

struct NovaToArithOp{
  template <typename OpTy>
  static Value maptop(OpTy op,Type resultType,ValueRange input,OpBuilder* builder){
    return mappingArith(op,resultType,input,builder);
  } 

  private:
  template <typename OpTy>
  static Value mappingArith(OpTy op,Type resultType,ValueRange input,OpBuilder* builder){
  return nullptr;
  }
  static Value mappingArith(nova::ConstantOp op,Type resultType,ValueRange input,OpBuilder* builder){
    return builder ->create<arith::ConstantOp>(op.getLoc(),resultType,op.getValue());
  }
};

// struct NovaConstantOpLowering 
//     : public OpConversionPattern<nova::ConstantOp> {
//   using OpConversionPattern<nova::ConstantOp>::OpConversionPattern;

//   LogicalResult matchAndRewrite(
//       nova::ConstantOp op, OpAdaptor adaptor,
//       ConversionPatternRewriter &rewriter) const override {
    
//     // Get the value attribute
//     Attribute valueAttr = op.getValue();
    
//     // // Check if it's a TypedAttr
//     auto typedAttr = dyn_cast<TypedAttr>(valueAttr);
//     if (!typedAttr) {
//       return rewriter.notifyMatchFailure(
//           op, "constant value must be a TypedAttr");
//     }
//     mlir::Type resultType = op.getResult().getType();
//     // Create the arith.constant operation
//     rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, typedAttr);
    
//     return success();
//   }
// };


//template
template <typename NovaArithOp>
class NovaArithConversionPattern : public OpConversionPattern<NovaArithOp>{
   public :
   using OpConversionPattern<NovaArithOp> :: OpConversionPattern;
   using OpAdaptor = typename NovaArithOp::Adaptor;

   LogicalResult matchAndRewrite(NovaArithOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    ValueRange operands = adaptor.getOperands();

    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if(!resultType)
    return rewriter.notifyMatchFailure(op,"Needs Tensor Type");

    Value result = NovaToArithOp::maptop(op,resultType,operands,&rewriter);
    if (!result)
      return rewriter.notifyMatchFailure(op, "Mapping to Arith failed or returned null");
    rewriter.replaceOp(op,result);
    return success();
   }
};


// Pass Definition

namespace {
struct NovaToArithLoweringPass
    : public PassWrapper<NovaToArithLoweringPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaToArithLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
  }

  StringRef getArgument() const final { return "convert-nova-to-arith"; }
  
  StringRef getDescription() const final {
    return "Lower Nova dialect operations to Arith dialect";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    ConversionTarget target(getContext());
    
    target.addLegalDialect<arith::ArithDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalOp<nova::ConstantOp>();

    RewritePatternSet patterns(&getContext());

    
    populateNovaToArithConversionPatterns(patterns);
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

void populateNovaToArithConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<NovaArithConversionPattern<nova::ConstantOp>>(
    patterns.getContext()
  );
}

std::unique_ptr<Pass> createNovaToArithLoweringPass() {
  return std::make_unique<NovaToArithLoweringPass>();
}

// Register the pass
void registerNovaToArithLoweringPass() {
  PassRegistration<NovaToArithLoweringPass>();
}

} // namespace nova
} // namespace mlir