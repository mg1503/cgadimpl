#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "Compiler/Translation/NovaToLinalg/NovaToLinalg.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Dialect/nova/NovaDialect.h"

namespace mlir
{
  namespace nova
  {

    // Conversion Patterns

    struct NovaBroadcastInDimOpLowering : public OpConversionPattern<nova::BroadcastInDimOp>
    {
      using OpConversionPattern<nova::BroadcastInDimOp>::OpConversionPattern;

      LogicalResult
      matchAndRewrite(nova::BroadcastInDimOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter) const override
      {

        Value input = adaptor.getOperand();

        auto resultType = llvm::dyn_cast<RankedTensorType>(op.getType());
        if (!resultType)
        {
          return rewriter.notifyMatchFailure(op, "expected ranked tensor result type");
        }

        auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
        if (!inputType)
        {
          return rewriter.notifyMatchFailure(op, "expected ranked tensor input type");
        }

        auto loc = op.getLoc();
        auto dimsAttr = op.getBroadcastDimensions();

        // Create empty output tensor
        Value emptyTensor = rewriter.create<tensor::EmptyOp>(
            loc, resultType.getShape(), resultType.getElementType());

        // Build affine map for input
        SmallVector<AffineExpr> inputExprs;
        for (auto [inputIdx, dimAttr] : llvm::enumerate(dimsAttr.getAsValueRange<IntegerAttr>()))
        {
          int64_t outputDim = dimAttr.getSExtValue();
          int64_t inputSize = inputType.getDimSize(inputIdx);
          int64_t outputSize = resultType.getDimSize(outputDim);

          // If broadcasting dimension (1 -> N), use constant 0
          if (inputSize == 1 && outputSize != 1)
          {
            inputExprs.push_back(rewriter.getAffineConstantExpr(0));
          }
          else
          {
            inputExprs.push_back(rewriter.getAffineDimExpr(outputDim));
          }
        }

        // Build affine map for output (identity)
        SmallVector<AffineExpr> outputExprs;
        for (unsigned i = 0; i < resultType.getRank(); ++i)
        {
          outputExprs.push_back(rewriter.getAffineDimExpr(i));
        }

        auto inputMap = AffineMap::get(resultType.getRank(), 0, inputExprs, rewriter.getContext());
        auto outputMap = AffineMap::get(resultType.getRank(), 0, outputExprs, rewriter.getContext());

        SmallVector<AffineMap> indexingMaps = {inputMap, outputMap};
        SmallVector<utils::IteratorType> iteratorTypes(resultType.getRank(),
                                                       utils::IteratorType::parallel);

        // Create linalg.generic for broadcast
        auto genericOp = rewriter.create<linalg::GenericOp>(
            loc,
            TypeRange{resultType},
            input, emptyTensor,
            indexingMaps,
            iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args)
            {
              b.create<linalg::YieldOp>(loc, args[0]);
            });

        rewriter.replaceOp(op, genericOp.getResults());
        return success();
      }
    };

    //-----------------------------------------------------------------------------
    // Matmul lowering
    //-----------------------------------------------------------------------------
    struct NovaMatmulOpLowering : public OpConversionPattern<nova::MatmulOp>
    {
      using OpConversionPattern<nova::MatmulOp>::OpConversionPattern;

      LogicalResult
      matchAndRewrite(nova::MatmulOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter) const override
      {

        auto operands = adaptor.getOperands();

        if (operands.size() != 2)
        {
          return rewriter.notifyMatchFailure(op, "expected exactly 2 operands");
        }

        Value lhs = operands[0];
        Value rhs = operands[1];

        // Get result type and create empty output tensor
        auto resultType = llvm::dyn_cast<RankedTensorType>(op.getType());
        if (!resultType)
        {
          return rewriter.notifyMatchFailure(op, "expected ranked tensor result");
        }
        // create a constnt zero
        Value cst = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getZeroAttr(resultType.getElementType()));
        // Create an empty tensor for the output
        Value emptyTensor = rewriter.create<tensor::EmptyOp>(
            op.getLoc(),
            resultType.getShape(),
            resultType.getElementType());
        // create a fill op to initialize the output tensor to zero
        Value outputTensor = rewriter.create<linalg::FillOp>(
                                         op.getLoc(), cst, emptyTensor)
                                 .getResult(0);

        // Create linalg.matmul with inputs and output
        rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
            op,
            ValueRange{lhs, rhs}, // inputs
            outputTensor);        // outputs
        return success();
      }
    };
    //-------------------------------------------------------------------
    // Square
    //-------------------------------------------------------------------

    // struct NovaSquareOpLowering : public OpConversionPattern<nova::SquareOp> {
    //   using OpConversionPattern<nova::SquareOp>::OpConversionPattern;

    //   LogicalResult matchAndRewrite(nova::SquareOp op,OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override{
    //     auto operands = adaptor.getOperands();

    //     Value lhs = operands [0];

    //     auto resultType = llvm::dyn_cast<RankedTensorType>(op.getType());
    //     if (!resultType) {
    //       return rewriter.notifyMatchFailure(op, "expected ranked tensor result type");
    //     }

    //     auto inputType = llvm::dyn_cast<RankedTensorType>(lhs.getType());
    //     if (!inputType) {
    //       return rewriter.notifyMatchFailure(op, "expected ranked tensor input type");
    //     }

    //     auto loc = op.getLoc();

    //     Value emptyTensor = rewriter.create<tensor::EmptyOp>(
    //         loc, resultType.getShape(), resultType.getElementType());

    //     rewriter.replaceOpWithNewOp<linalg::SquareOp>(
    //         op,
    //         ValueRange{lhs},      // inputs
    //         emptyTensor); // outputs
    //     return success();

    //   }
    //};

    void populateNovaToLinalgPatterns(RewritePatternSet &patterns)
    {
      patterns.add<NovaMatmulOpLowering,
                   NovaBroadcastInDimOpLowering
                   //    ,NovaSquareOpLowering
                   >(patterns.getContext());
    }
  } // namespace nova
} // namespace mlir