// =====================
// file: cgadimpl/src/core/mlir_emitter.cpp
// MLIR C++ API implementation for direct graph emission
// =====================
#include "ad/core/mlir_emitter.hpp"
#include "ad/core/graph.hpp"
#include "ad/core/schema.hpp"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>

namespace ag::jit {

MLIREmitter::MLIREmitter() {
    context_ = std::make_shared<mlir::MLIRContext>();
    registerDialects();
}

MLIREmitter::~MLIREmitter() = default;

void MLIREmitter::registerDialects() {
    // The emitter only needs Nova and Func dialects to build the initial IR.
    // Optimization and bufferization are now handled within NovaCompilerAPI.
    context_->getOrLoadDialect<mlir::nova::NovaDialect>();
    context_->getOrLoadDialect<mlir::func::FuncDialect>();
}

mlir::Type MLIREmitter::dtypeToMLIRType(mlir::OpBuilder& builder, OwnTensor::Dtype dtype) {
    switch (dtype) {
        case OwnTensor::Dtype::Float32:  return builder.getF32Type();
        case OwnTensor::Dtype::Float16:  return builder.getF16Type();
        case OwnTensor::Dtype::Bfloat16: return builder.getBF16Type();
        case OwnTensor::Dtype::Int32:    return builder.getI32Type();
        case OwnTensor::Dtype::Int64:    return builder.getI64Type();
        default:
            llvm::errs() << "Unsupported dtype in MLIR emission\n";
            return builder.getF32Type(); // fallback
    }
}

mlir::RankedTensorType MLIREmitter::createTensorType(
    mlir::OpBuilder& builder,
    const std::vector<int64_t>& shape,
    OwnTensor::Dtype dtype
) {
    auto elemType = dtypeToMLIRType(builder, dtype);
    return mlir::RankedTensorType::get(shape, elemType);
}

std::pair<mlir::OwningOpRef<mlir::ModuleOp>, std::string> 
MLIREmitter::emitModule(const Plan& plan) {
    mlir::OpBuilder builder(context_.get());
    auto loc = builder.getUnknownLoc();

    // Create module
    auto module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());

    // Build function signature
    llvm::SmallVector<mlir::Type, 8> inputTypes;
    
    // Add input tensors to function signature
    for (const auto& meta : plan.sig.in_meta) {
        inputTypes.push_back(createTensorType(builder, meta.shape, meta.dtype));
    }
    
    // Add parameter tensors to function signature
    for (const auto& meta : plan.sig.param_meta) {
        inputTypes.push_back(createTensorType(builder, meta.shape, meta.dtype));
    }

    // Output type comes from the last step
    if (plan.steps.empty()) {
        llvm::errs() << "Error: Empty plan in MLIR emission\n";
        return {mlir::OwningOpRef<mlir::ModuleOp>(module), ""};
    }

    const auto& output_meta = plan.steps.back().out_meta;
    auto output_shape = output_meta.shape;
    
    // Total reduction rank adjustment: if it's a total reduction (Sum/MeanAll), it returns rank 0
    if ((plan.steps.back().op == Op::Sum || plan.steps.back().op == Op::MeanAll) && 
        output_shape.size() == 1 && output_shape[0] == 1) {
        output_shape = {};
    }
    
    auto outputType = createTensorType(builder, output_shape, output_meta.dtype);

    // Create function type
    auto funcType = builder.getFunctionType(inputTypes, outputType);

    // Create function
    auto func = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
    auto& entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    // Map slots to MLIR values
    llvm::DenseMap<int, mlir::Value> slotMap;

    // Map function arguments to their sources
    size_t argIdx = 0;
    
    // First, map all inputs
    for (size_t i = 0; i < plan.sig.in_meta.size(); ++i) {
        // Inputs will be referenced via ArgInput in steps
        // We'll handle them during step processing
        argIdx++;
    }
    
    // Then params
    for (size_t i = 0; i < plan.sig.param_meta.size(); ++i) {
        argIdx++;
    }

    // Helper to get MLIR value from Arg
    auto getValueForArg = [&](const Arg& arg) -> mlir::Value {
        return std::visit([&](auto&& a) -> mlir::Value {
            using T = std::decay_t<decltype(a)>;
            if constexpr (std::is_same_v<T, ArgInput>) {
                return entryBlock.getArgument(a.idx);
            } else if constexpr (std::is_same_v<T, ArgParam>) {
                return entryBlock.getArgument(plan.sig.in_meta.size() + a.idx);
            } else if constexpr (std::is_same_v<T, ArgSlot>) {
                return slotMap[a.slot];
            } else if constexpr (std::is_same_v<T, ArgLit>) {
                // For literals, we create a nova.constant op
                const auto& t = a.t;
                auto shape = t.shape().dims;
                auto type = createTensorType(builder, shape, t.dtype());
                
                // For now, support float32 literals
                if (t.dtype() == OwnTensor::Dtype::Float32) {
                    auto data = t.to_cpu();
                    auto attr = mlir::DenseElementsAttr::get(type, llvm::ArrayRef<float>(static_cast<const float*>(data.data()), t.numel()));
                    return builder.create<mlir::nova::ConstantOp>(loc, type, attr).getResult();
                } else {
                    llvm::errs() << "Error: Unsupported literal dtype in MLIR emission\n";
                    return mlir::Value();
                }
            }
        }, arg);
    };

    // Process each step
    for (const auto& step : plan.steps) {
        llvm::SmallVector<mlir::Value, 4> operands;
        
        // Gather operands
        for (const auto& arg : step.args) {
            auto val = getValueForArg(arg);
            if (!val) {
                llvm::errs() << "Error: Failed to get value for argument\n";
                return {mlir::OwningOpRef<mlir::ModuleOp>(module), ""};
            }
            operands.push_back(val);
        }

        // Create the appropriate Nova operation
        mlir::Value result;
        auto resultType = createTensorType(builder, step.out_meta.shape, step.out_meta.dtype);

        switch (step.op) {
            case Op::Add:
                if (operands.size() == 2) {
                    result = builder.create<mlir::nova::AddOp>(
                        loc, resultType, operands[0], operands[1]
                    ).getResult();
                }
                break;

            case Op::Sub:
                if (operands.size() == 2) {
                    result = builder.create<mlir::nova::SubOp>(
                        loc, resultType, operands[0], operands[1]
                    ).getResult();
                }
                break;

            case Op::Mul:
                if (operands.size() == 2) {
                    result = builder.create<mlir::nova::MulOp>(
                        loc, resultType, operands[0], operands[1]
                    ).getResult();
                }
                break;

            case Op::MatMul:
                if (operands.size() == 2) {
                    result = builder.create<mlir::nova::MatmulOp>(
                        loc, resultType, operands[0], operands[1]
                    ).getResult();
                }
                break;

            case Op::Exp:
                if (operands.size() == 1) {
                    result = builder.create<mlir::nova::ExpOp>(
                        loc, resultType, operands[0]
                    ).getResult();
                }
                break;

            case Op::Log:
                if (operands.size() == 1) {
                    result = builder.create<mlir::nova::LogOp>(
                        loc, resultType, operands[0]
                    ).getResult();
                }
                break;

            case Op::Tanh:
                if (operands.size() == 1) {
                    result = builder.create<mlir::nova::TanhOp>(
                        loc, resultType, operands[0]
                    ).getResult();
                }
                break;

            case Op::Relu:
                if (operands.size() == 1) {
                    result = builder.create<mlir::nova::ReluOp>(
                        loc, resultType, operands[0]
                    ).getResult();
                }
                break;

            case Op::Sum:
                if (operands.size() == 1) {
                    // Sum all dimensions - Nova expects rank-0 for total reduction if keepdims=false
                    auto scalarType = mlir::RankedTensorType::get({}, resultType.getElementType());
                    result = builder.create<mlir::nova::ReduceOp>(
                        loc,
                        mlir::nova::ReductionKind::SUM,
                        operands[0],
                        scalarType,
                        /*keepdims=*/false,
                        /*dimension=*/llvm::ArrayRef<int64_t>{},
                        /*ignore_nan=*/false
                    ).getResult();
                }
                break;

            case Op::RowSum:
                if (operands.size() == 1) {
                    // Reduce along axis 1
                    result = builder.create<mlir::nova::ReduceOp>(
                        loc,
                        mlir::nova::ReductionKind::SUM,
                        operands[0],
                        resultType,
                        /*keepdims=*/true,
                        llvm::ArrayRef<int64_t>{1},
                        /*ignore_nan=*/false
                    ).getResult();
                }
                break;

            case Op::RowMax:
                if (operands.size() == 1) {
                    // Reduce along axis 1
                    result = builder.create<mlir::nova::ReduceOp>(
                        loc,
                        mlir::nova::ReductionKind::MAX,
                        operands[0],
                        resultType,
                        /*keepdims=*/true,
                        llvm::ArrayRef<int64_t>{1},
                        /*ignore_nan=*/false
                    ).getResult();
                }
                break;

            case Op::MeanAll:
                if (operands.size() == 1) {
                    auto scalarType = mlir::RankedTensorType::get({}, resultType.getElementType());
                    result = builder.create<mlir::nova::ReduceOp>(
                        loc,
                        mlir::nova::ReductionKind::MEAN,
                        operands[0],
                        scalarType,
                        /*keepdims=*/false,
                        llvm::ArrayRef<int64_t>{},
                        /*ignore_nan=*/false
                    ).getResult();
                }
                break;

            default:
                llvm::errs() << "Warning: Unsupported op " << op_name(step.op) 
                           << " in MLIR emission, skipping\n";
                // For unsupported ops, we can't continue
                return {mlir::OwningOpRef<mlir::ModuleOp>(module), ""};
        }

        if (!result) {
            llvm::errs() << "Error: Failed to create MLIR op for " << op_name(step.op) << "\n";
            return {mlir::OwningOpRef<mlir::ModuleOp>(module), ""};
        }

        // Store result in slot map
        slotMap[step.out_slot] = result;
    }

    // Create return statement
    auto returnValue = slotMap[plan.out_slot];
    if (!returnValue) {
        llvm::errs() << "Error: Output slot not found in MLIR emission\n";
        return {mlir::OwningOpRef<mlir::ModuleOp>(module), ""};
    }

    builder.create<mlir::func::ReturnOp>(loc, returnValue);

    // Verify the module
    if (mlir::failed(mlir::verify(module))) {
        llvm::errs() << "Error: MLIR module verification failed\n";
        module.dump();
        return {mlir::OwningOpRef<mlir::ModuleOp>(module), ""};
    }

    // Serialize to string
    std::string mlirStr;
    llvm::raw_string_ostream os(mlirStr);
    module.print(os);
    os.flush();

    return {mlir::OwningOpRef<mlir::ModuleOp>(module), mlirStr};
}

} // namespace ag::jit
