

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "Compiler/Translation/NovaToTosa/NovaToTosa.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include "Compiler/Dialect/nova/NovaDialect.h"
namespace mlir
{
  namespace nova
  {

    // functions which will be called inside template
    struct NovaOpTosaOp
    {
      // helper function
      static SmallVector<int64_t> shapeFind(Type currType, int64_t axis)
      {
        SmallVector<int64_t> newshape; // paramters=>inputshape(auto) and axis(int32)
        auto rankedType = cast<RankedTensorType>(currType);
        for (int64_t i = 0; i < rankedType.getRank(); ++i)
        {
          if (i == axis)
          {
            newshape.push_back(1); // TOSA keeps reduced dimension as size 1
          }
          else
          {
            newshape.push_back(rankedType.getDimSize(i));
          }
        }
        return newshape;
      }
      static SmallVector<int64_t> shapeFindargmax(Type currType, int64_t axis)
      {
        SmallVector<int64_t> newshape; // paramters=>inputshape(auto) and axis(int32)
        auto rankedType = cast<RankedTensorType>(currType);
        for (int64_t i = 0; i < rankedType.getRank(); ++i)
        {
          if (i == axis)
          {
            // newshape.push_back(1); // TOSA keeps reduced dimension as size 1
          }
          else
          {
            newshape.push_back(rankedType.getDimSize(i));
          }
        }
        return newshape;
      }
      // converting two operand to boolean function

      // collect the operand data type
      // if float bitcast to equivalent int
      //  use trunc to lower int from any bitwidth to i1.
      // static  Value reducetobool(Location loc,Value rhs,Value rhs,OpBuilder* builder){
      // auto lhstype=lhs.getType();
      // auto rhstype=rhs.getType();
      // //converting first operand
      // if(isa<FloatType>(rhstype)){
      //   auto rhsbw = frhstype.getWidth();

      //   auto v = builder ->create<tosa::CastOp>(loc,builder->get)
      // }
      // converting second operand

      static int64_t shapeFindforargmax(Type currType)
      {
        int64_t newshape = 1; // paramters=>inputshape(auto) and axis(int32)
        auto rankedType = cast<RankedTensorType>(currType);
        for (int64_t i = 0; i < rankedType.getRank(); ++i)
        {
          newshape *= rankedType.getDimSize(i);
        }
        return newshape;
      }
      template <typename OpTy>
      static Value maptop(OpTy op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        return mappingtosa(op, resultType, input, builder);
      }

    private:
      template <typename OpTy>
      static Value mappingtosa(OpTy op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        return nullptr;
      }
      static Value mappingtosa(nova::MaxOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {

        auto restensor = dyn_cast<mlir::TensorType>(resultType);
        auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
        auto w = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[1]);

        return builder->create<tosa::MaximumOp>(op.getLoc(), resultType, v, w);
      }
      static Value mappingtosa(nova::MinOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        auto restensor = dyn_cast<mlir::TensorType>(resultType);
        auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
        auto w = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[1]);
        return builder->create<tosa::MinimumOp>(op.getLoc(), resultType, v, w);
      }
      static Value mappingtosa(nova::AndOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        auto restensor = dyn_cast<mlir::TensorType>(resultType);
        auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
        auto w = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[1]);
        return builder->create<tosa::LogicalAndOp>(op.getLoc(), resultType, v, w);
      }
      static Value mappingtosa(nova::OrOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        auto restensor = dyn_cast<mlir::TensorType>(resultType);
        auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
        auto w = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[1]);

        return builder->create<tosa::LogicalOrOp>(op.getLoc(), resultType, v, w);
      }

      static Value mappingtosa(nova::XorOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        auto restensor = dyn_cast<mlir::TensorType>(resultType);
        auto v = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[0]);
        auto w = builder->create<tosa::CastOp>(op.getLoc(), restensor, input[1]);

        return builder->create<tosa::LogicalXorOp>(op.getLoc(), resultType, v, w);
      }
      static Value mappingtosa(nova::NegOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        return builder->create<tosa::NegateOp>(op.getLoc(), resultType, input[0]);
      }
      static Value mappingtosa(nova::ReciprocalOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        return builder->create<tosa::ReciprocalOp>(op.getLoc(), resultType, input[0]);
      }
      static Value mappingtosa(nova::NotOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        auto restype = dyn_cast<mlir::TensorType>(resultType);
        auto v = builder->create<tosa::CastOp>(op.getLoc(), restype, input[0]);

        return builder->create<tosa::LogicalNotOp>(op.getLoc(), resultType, v);
      }
      //-------------------------reduce op----------------------
      static Value mappincasereduce(nova::ReduceOp op, Type temresult, Value v, mlir::IntegerAttr axisAttr, OpBuilder *builder, mlir::tosa::NanPropagationModeAttr nanmode)
      {
        nova::ReductionKind rk = op.getKind();
        switch (rk)
        {
        case nova::ReductionKind::MAX:
          return builder->create<tosa::ReduceMaxOp>(op.getLoc(), temresult, v, axisAttr, nanmode);
        case nova::ReductionKind::MIN:
          return builder->create<tosa::ReduceMinOp>(op.getLoc(), temresult, v, axisAttr, nanmode);
        case nova::ReductionKind::PRODUCT:
          return builder->create<tosa::ReduceProductOp>(op.getLoc(), temresult, v, axisAttr);
        case nova::ReductionKind::SUM:
          return builder->create<tosa::ReduceSumOp>(op.getLoc(), temresult, v, axisAttr);

        case nova::ReductionKind::MEAN:
        {
          auto sum = builder->create<tosa::ReduceSumOp>(op.getLoc(), temresult, v, axisAttr);
          int64_t axis = axisAttr.getInt();
          auto inputType = cast<RankedTensorType>(v.getType());
          int64_t dimSize = inputType.getDimSize(axis);

          Value divisor;
          auto elementType = inputType.getElementType();

          if (inputType.isDynamicDim(axis))
          {
            Value dimVal = builder->create<tensor::DimOp>(op.getLoc(), v, axis);
            if (isa<FloatType>(elementType))
            {
              divisor = builder->create<mlir::arith::IndexCastOp>(op.getLoc(), builder->getI64Type(), dimVal);
              divisor = builder->create<mlir::arith::UIToFPOp>(op.getLoc(), elementType, divisor);
            }
            else
            {
              divisor = builder->create<mlir::arith::IndexCastOp>(op.getLoc(), elementType, dimVal);
            }
          }
          else
          {
            if (isa<FloatType>(elementType))
            {
              divisor = builder->create<tosa::ConstOp>(op.getLoc(),
                                                       RankedTensorType::get({}, elementType),
                                                       DenseElementsAttr::get(RankedTensorType::get({}, elementType),
                                                                              builder->getFloatAttr(elementType, static_cast<double>(dimSize))));
            }
            else
            {
              divisor = builder->create<tosa::ConstOp>(op.getLoc(),
                                                       RankedTensorType::get({}, elementType),
                                                       DenseElementsAttr::get(RankedTensorType::get({}, elementType),
                                                                              builder->getIntegerAttr(elementType, dimSize)));
            }
          }

          // Reshape divisor to match rank of sum for broadcasting
          auto resultType = cast<RankedTensorType>(temresult);
          int64_t rank = resultType.getRank();
          SmallVector<int64_t> newShape(rank, 1);

          auto shapeType = RankedTensorType::get({rank}, builder->getIndexType());
          auto shapeAttr = DenseIntElementsAttr::get(shapeType, newShape);
          auto shapeConst = builder->create<tosa::ConstShapeOp>(op.getLoc(),
                                                                mlir::tosa::shapeType::get(builder->getContext(), rank),
                                                                shapeAttr);

          auto reshapedDivisorType = RankedTensorType::get(newShape, elementType);
          auto reshapedDivisor = builder->create<tosa::ReshapeOp>(op.getLoc(), reshapedDivisorType, divisor, shapeConst);

          if (isa<FloatType>(elementType))
          {
            auto reciprocal = builder->create<tosa::ReciprocalOp>(op.getLoc(), reshapedDivisorType, reshapedDivisor);
            auto shift = builder->create<tosa::ConstOp>(op.getLoc(),
                                                        RankedTensorType::get({1}, builder->getI8Type()),
                                                        DenseElementsAttr::get(RankedTensorType::get({1}, builder->getI8Type()),
                                                                               builder->getI8IntegerAttr(0)));
            return builder->create<tosa::MulOp>(op.getLoc(), temresult, sum, reciprocal, shift);
          }
          else
          {
            return builder->create<tosa::IntDivOp>(op.getLoc(), temresult, sum, reshapedDivisor);
          }
        }
        case nova::ReductionKind::ALL:
          return builder->create<tosa::ReduceAllOp>(op.getLoc(), temresult, v, axisAttr);
        case nova::ReductionKind::ANY:
          return builder->create<tosa::ReduceAnyOp>(op.getLoc(), temresult, v, axisAttr);
        }
        return nullptr;
      }

      //-------------------------Reduce-ArgMax---------------------------

      static Value mappingtosa(nova::ArgmaxOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        Value v = input[0]; // the final result is store in this
        auto inputType = dyn_cast<RankedTensorType>(v.getType());
        auto resultdt = dyn_cast<RankedTensorType>(resultType);
        auto ignorenanAttr = op.getIgnoreNan();
        mlir::tosa::NanPropagationModeAttr nanmode;
        nanmode = mlir::tosa::NanPropagationModeAttr::get(op->getContext(), mlir::tosa::NanPropagationMode::PROPAGATE);

        if (ignorenanAttr)
        {
          nanmode = mlir::tosa::NanPropagationModeAttr::get(op->getContext(), mlir::tosa::NanPropagationMode::IGNORE);
        }
        // getting dimension
        auto dimensionAttr = op.getDimension();
        // reducing with dimesion-----------
        if (dimensionAttr.has_value())
        {
          int64_t axisValue = dimensionAttr.value();
          // getting value for axis attribute
          auto axisAttr = builder->getI32IntegerAttr(axisValue);
          // cretaing result tensor
          auto tempshape = shapeFindargmax(inputType, axisValue);
          auto temptype = RankedTensorType::get(tempshape, resultdt.getElementType());

          // we have to replac the resulttype
          v = builder->create<tosa::ArgMaxOp>(op.getLoc(), temptype, v, axisAttr, nanmode);
          //  op.emitOpError("dimension attribute missing for TOSA mapping");
        }
        // No dimension - reduce all dimension
        else
        {
          auto finalShape = shapeFindforargmax(v.getType());
          // Create the final result type
          auto finalType = RankedTensorType::get({}, resultdt.getElementType());
          auto shapeTensorType = RankedTensorType::get(
              {1}, builder->getIndexType());
          auto shapeAttr = DenseIntElementsAttr::get(
              shapeTensorType,
              finalShape);
          // flatten it and
          //  Create const op
          Value shapeValue = builder->create<tosa::ConstShapeOp>(
              op.getLoc(), mlir::tosa::shapeType::get(builder->getContext(), 1), shapeAttr);
          // Perform reshape
          Value reshapedres = builder->create<tosa::ReshapeOp>(
              op.getLoc(), v, shapeValue);
          auto axisAttr = builder->getI32IntegerAttr(0);
          v = builder->create<tosa::ArgMaxOp>(op.getLoc(), finalType, reshapedres, axisAttr, nanmode);
        }
        // KEEP DIMS
        if (op.getKeepdims())
        {
          auto finalShape = resultdt.getShape();

          // Create the final result type
          auto finalType = RankedTensorType::get(finalShape, resultdt.getElementType());
          auto shapeTensorType = RankedTensorType::get(
              {static_cast<int64_t>(finalShape.size())}, builder->getIndexType());

          SmallVector<int64_t> shapeValues(finalShape.begin(), finalShape.end());

          auto shapeAttr = DenseIntElementsAttr::get(
              shapeTensorType,
              llvm::ArrayRef(shapeValues.data(), shapeValues.size()));

          // Create const op
          Value shapeValue = builder->create<tosa::ConstShapeOp>(
              op.getLoc(), mlir::tosa::shapeType::get(builder->getContext(), finalShape.size()), shapeAttr);

          // Perform reshape
          v = builder->create<tosa::ReshapeOp>(
              op.getLoc(), finalType, v, shapeValue);
        }

        return v;
      }
      //----------------------------------------ARGMIN-----------------------------

      static Value mappingtosa(nova::ArgMinOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        Value v = input[0]; // the final result is store in this
        auto inputType = dyn_cast<RankedTensorType>(v.getType());
        auto resultdt = dyn_cast<RankedTensorType>(resultType);
        auto ignorenanAttr = op.getIgnoreNan();
        mlir::tosa::NanPropagationModeAttr nanmode;
        nanmode = mlir::tosa::NanPropagationModeAttr::get(op->getContext(), mlir::tosa::NanPropagationMode::PROPAGATE);

        if (ignorenanAttr)
        {
          nanmode = mlir::tosa::NanPropagationModeAttr::get(op->getContext(), mlir::tosa::NanPropagationMode::IGNORE);
        }
        v = builder->create<tosa::NegateOp>(op.getLoc(), v.getType(), v);

        // getting dimension
        auto dimensionAttr = op.getDimension();
        // reducing with dimesion-----------
        if (dimensionAttr.has_value())
        {
          int64_t axisValue = dimensionAttr.value();
          // getting value for axis attribute
          auto axisAttr = builder->getI32IntegerAttr(axisValue);
          // cretaing result tensor
          auto tempshape = shapeFindargmax(inputType, axisValue);
          auto temptype = RankedTensorType::get(tempshape, resultdt.getElementType());

          // we have to replac the resulttype
          v = builder->create<tosa::ArgMaxOp>(op.getLoc(), temptype, v, axisAttr, nanmode);
          //  op.emitOpError("dimension attribute missing for TOSA mapping");
        }
        // No dimension - reduce all dimension
        else
        {
          auto finalShape = shapeFindforargmax(v.getType());
          // Create the final result type
          auto finalType = RankedTensorType::get({}, resultdt.getElementType());
          auto shapeTensorType = RankedTensorType::get(
              {1}, builder->getIndexType());
          auto shapeAttr = DenseIntElementsAttr::get(
              shapeTensorType,
              finalShape);
          // flatten it and
          //  Create const op
          Value shapeValue = builder->create<tosa::ConstShapeOp>(
              op.getLoc(), mlir::tosa::shapeType::get(builder->getContext(), 1), shapeAttr);
          // Perform reshape
          Value reshapedres = builder->create<tosa::ReshapeOp>(
              op.getLoc(), v, shapeValue);
          auto axisAttr = builder->getI32IntegerAttr(0);
          v = builder->create<tosa::ArgMaxOp>(op.getLoc(), finalType, reshapedres, axisAttr, nanmode);
        }
        // KEEP DIMS
        if (op.getKeepdims())
        {
          auto finalShape = resultdt.getShape();

          // Create the final result type
          auto finalType = RankedTensorType::get(finalShape, resultdt.getElementType());
          auto shapeTensorType = RankedTensorType::get(
              {static_cast<int64_t>(finalShape.size())}, builder->getIndexType());

          SmallVector<int64_t> shapeValues(finalShape.begin(), finalShape.end());

          auto shapeAttr = DenseIntElementsAttr::get(
              shapeTensorType,
              llvm::ArrayRef(shapeValues.data(), shapeValues.size()));

          // Create const op
          Value shapeValue = builder->create<tosa::ConstShapeOp>(
              op.getLoc(), mlir::tosa::shapeType::get(builder->getContext(), finalShape.size()), shapeAttr);
          // Perform reshape
          v = builder->create<tosa::ReshapeOp>(
              op.getLoc(), finalType, v, shapeValue);
        }

        return v;
      }
      //---------------------------Reduce-Op---------------------------------

      static Value mappingtosa(nova::ReduceOp op, Type resultType, ValueRange input, OpBuilder *builder)
      {
        Value v = input[0]; // the final result is store in this
        // getting the axis from dimension
        auto dimensionAttr = op.getDimension();
        auto inputType = dyn_cast<RankedTensorType>(v.getType());
        auto result1Type = dyn_cast<RankedTensorType>(resultType);
        // setting ignore nan attribute to nan mode
        auto ignorenanAttr = op.getIgnoreNan();
        mlir::tosa::NanPropagationModeAttr nanmode;
        nanmode = mlir::tosa::NanPropagationModeAttr::get(op->getContext(), mlir::tosa::NanPropagationMode::PROPAGATE);

        if (ignorenanAttr)
        {
          nanmode = mlir::tosa::NanPropagationModeAttr::get(op->getContext(), mlir::tosa::NanPropagationMode::IGNORE);
        }

        // 🪻👍🏻🪻
        // reducing with dimesion-----------
        if (dimensionAttr.has_value())
        {
          auto dimension = dimensionAttr.value();
          for (auto dim : dimension)
          {
            // getting value for axis attribute
            int64_t axisValue = dyn_cast<IntegerAttr>(dim).getInt();
            auto axisAttr = builder->getI32IntegerAttr(axisValue);
            // getting temp shape
            auto tempshape = shapeFind(v.getType(), axisValue); // placeholder for now
            auto currType = cast<RankedTensorType>(v.getType());
            auto tempresult = RankedTensorType::get(tempshape, currType.getElementType());
            // getting the correct operation
            v = mappincasereduce(op, tempresult, v, axisAttr, builder, nanmode);
          }
        }
        // No dimension - reduce all dimension
        else
        {
          auto inputRank = inputType.getRank();
          for (int64_t axis = inputRank - 1; axis >= 0; --axis)
          {
            auto axisAttr = builder->getI32IntegerAttr(axis);
            auto tempShape = shapeFind(v.getType(), axis);
            auto currType = cast<RankedTensorType>(v.getType());
            auto tempresult = RankedTensorType::get(tempShape, currType.getElementType());
            v = mappincasereduce(op, tempresult, v, axisAttr, builder, nanmode);
          }
        }
        // NEED TO ADD KEEP DIMS HERE
        if (!op.getKeepdims())
        {
          auto currentType = cast<RankedTensorType>(v.getType());
          auto finalShape = result1Type.getShape();

          // Create the final result type
          auto finalType = RankedTensorType::get(finalShape, currentType.getElementType());
          auto shapeTensorType = RankedTensorType::get(
              {static_cast<int64_t>(finalShape.size())}, builder->getIndexType());

          SmallVector<int64_t> shapeValues(finalShape.begin(), finalShape.end());

          auto shapeAttr = DenseIntElementsAttr::get(
              shapeTensorType,
              llvm::ArrayRef(shapeValues.data(), shapeValues.size()));

          // Create const op
          Value shapeValue = builder->create<tosa::ConstShapeOp>(
              op.getLoc(), mlir::tosa::shapeType::get(builder->getContext(), finalShape.size()), shapeAttr);

          // Perform reshape
          v = builder->create<tosa::ReshapeOp>(
              op.getLoc(), finalType, v, shapeValue);
        }

        return v;
      }
    };

    // Pattern to convert nova.relu to tosa.relu
    struct NovaReluOpLowering : public OpConversionPattern<ReluOp>
    {
      using OpConversionPattern<ReluOp>::OpConversionPattern;

      LogicalResult
      matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                      ConversionPatternRewriter &rewriter) const override
      {
        Location loc = op.getLoc();
        Value input = adaptor.getInput();
        auto inputType = cast<RankedTensorType>(input.getType());
        Type elementType = inputType.getElementType();

        // Create zero constant tensor with the same shape as input
        Attribute zeroAttr;

        if (auto floatType = dyn_cast<FloatType>(elementType))
        {
          APFloat zeroVal = APFloat::getZero(floatType.getFloatSemantics());
          zeroAttr = rewriter.getFloatAttr(floatType, zeroVal);
        }
        else if (auto intType = dyn_cast<IntegerType>(elementType))
        {
          zeroAttr = rewriter.getIntegerAttr(intType, 0);
        }
        else
        {
          return failure();
        }
        DenseElementsAttr zeroTensor = DenseElementsAttr::get(inputType, zeroAttr);
        Value zero = rewriter.create<nova::ConstantOp>(loc, inputType, zeroTensor);
        Value result = rewriter.create<tosa::MaximumOp>(
            loc, inputType, input, zero);

        rewriter.replaceOp(op, result);
        return success();
      }
    };
    //
    // creating a template
    template <typename NovaTopTy>
    class NovaToTosaLoweringTemplate : public OpConversionPattern<NovaTopTy>
    {
    public:
      using OpConversionPattern<NovaTopTy>::OpConversionPattern;
      using OpAdaptor = typename NovaTopTy::Adaptor; // for getting all meta data dynamically using adaptor
      LogicalResult matchAndRewrite(NovaTopTy op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override
      {
        ValueRange operands = adaptor.getOperands();
        // checking operand is empty or not
        if (operands.empty())
          return rewriter.notifyMatchFailure(op, "expected operands for tosa lowering operations");
        // getting resultType
        auto resultType = op.getResult().getType();
        // if (!mlir::isa<RankedTensorType>(resultType))
        //   return rewriter.notifyMatchFailure(op, "expected ranked tensor result");
        Value result = NovaOpTosaOp::maptop(
            op, resultType, operands, &rewriter);
        if (!result)
          return rewriter.notifyMatchFailure(op, "failed to map to TOSA operation");

        rewriter.replaceOp(op, result);
        return success();
      }
    };

    // pass definition
    namespace
    {
      struct NovaToTosaLoweringPass
          : public PassWrapper<NovaToTosaLoweringPass, OperationPass<ModuleOp>>
      {

        MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NovaToTosaLoweringPass)

        void getDependentDialects(DialectRegistry &registry) const override
        {
          registry.insert<tosa::TosaDialect>();
          registry.insert<func::FuncDialect>();
          registry.insert<nova::NovaDialect>();
        }

        StringRef getArgument() const final { return "convert-nova-to-tosa"; }

        StringRef getDescription() const final
        {
          return "Lower Nova dialect operations to Tosa dialect";
        }

        void runOnOperation() override
        {
          ModuleOp module = getOperation();
          ConversionTarget target(getContext());

          target.addLegalDialect<tosa::TosaDialect, func::FuncDialect>();
          target.addLegalOp<nova::ConstantOp>();
          target.addIllegalOp<nova::ReluOp>();
          target.addIllegalOp<nova::MaxOp>();
          target.addIllegalOp<nova::MinOp>();
          target.addIllegalOp<nova::AndOp>();
          target.addIllegalOp<nova::OrOp>();
          target.addIllegalOp<nova::XorOp>();
          target.addIllegalOp<nova::NegOp>();
          target.addIllegalOp<nova::NotOp>();
          target.addIllegalOp<nova::ReciprocalOp>();
          target.addIllegalOp<nova::ReduceOp>();
          target.addIllegalOp<nova::ArgmaxOp>();
          target.addIllegalOp<nova::ArgMinOp>();

          TypeConverter typeConverter;
          typeConverter.addConversion([](Type type)
                                      { return type; });
          RewritePatternSet patterns(&getContext());
          populateNovaToTosaConversionPatterns(patterns);

          if (failed(applyPartialConversion(module, target, std::move(patterns))))
          {
            signalPassFailure();
            return;
          }
        }
      };

    }

    void populateNovaToTosaConversionPatterns(RewritePatternSet &patterns)
    {
      patterns.add<NovaReluOpLowering,
                   NovaToTosaLoweringTemplate<nova::MaxOp>,
                   NovaToTosaLoweringTemplate<nova::MinOp>,
                   NovaToTosaLoweringTemplate<nova::AndOp>,
                   NovaToTosaLoweringTemplate<nova::OrOp>,
                   NovaToTosaLoweringTemplate<nova::XorOp>,
                   NovaToTosaLoweringTemplate<nova::NotOp>,
                   NovaToTosaLoweringTemplate<nova::NegOp>,
                   NovaToTosaLoweringTemplate<nova::ReciprocalOp>,
                   NovaToTosaLoweringTemplate<nova::ReduceOp>,
                   NovaToTosaLoweringTemplate<nova::ArgmaxOp>,
                   NovaToTosaLoweringTemplate<nova::ArgMinOp>

                   >(
          patterns.getContext());
    }

    // creating a pointer for this pass
    std::unique_ptr<Pass> createNovaToTosaLoweringPass()
    {
      return std::make_unique<NovaToTosaLoweringPass>();
    }

    // Register the pass
    void registerNovaToTosaLoweringPass()
    {
      PassRegistration<NovaToTosaLoweringPass>();
    }

  } // namespace nova
} // namespace mlir