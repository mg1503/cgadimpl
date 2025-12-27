// =====================
// file: cgadimpl/include/ad/mlir_emitter.hpp
// MLIR C++ API utilities for direct graph emission
// =====================
#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Compiler/Dialect/nova/NovaDialect.h"
#include "Compiler/Dialect/nova/NovaOps.h"
#include <memory>
#include <vector>
#include <string>
#include <variant>
#include "tensor.hpp"
#include "ad/core/schema.hpp"  // For Op enum and op_name

namespace ag {

// Forward declarations
struct Node;
struct Value;

namespace jit {

// Tensor metadata for MLIR emission
struct TensorMetadata {
    std::vector<int64_t> shape;
    OwnTensor::Dtype dtype;
    OwnTensor::DeviceIndex device;
};

// Signature of compiled function
struct Signature {
    std::vector<TensorMetadata> in_meta;
    std::vector<TensorMetadata> param_meta;
    
    bool matches(const std::vector<OwnTensor::Tensor*>& inputs,
                 const std::vector<OwnTensor::Tensor*>& params) const {
        if (inputs.size() != in_meta.size() || params.size() != param_meta.size()) {
            return false;
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i]->shape().dims != in_meta[i].shape ||
                inputs[i]->dtype() != in_meta[i].dtype ||
                inputs[i]->device().device != in_meta[i].device.device ||
                inputs[i]->device().index != in_meta[i].device.index) {
                return false;
            }
        }
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i]->shape().dims != param_meta[i].shape ||
                params[i]->dtype() != param_meta[i].dtype ||
                params[i]->device().device != param_meta[i].device.device ||
                params[i]->device().index != param_meta[i].device.index) {
                return false;
            }
        }
        return true;
    }
};

// Argument sources for a Step
struct ArgInput  { int idx; };
struct ArgParam  { int idx; };
struct ArgSlot   { int slot; };
struct ArgLit    { OwnTensor::Tensor t{OwnTensor::Shape{}, OwnTensor::Dtype::Float32}; };
using Arg = std::variant<ArgInput, ArgParam, ArgSlot, ArgLit>;

// Step in execution plan
struct Step {
    Op op;
    std::vector<Arg> args;
    int out_slot{};
    TensorMetadata out_meta;
};

// Compilation plan
struct Plan {
    Signature sig;
    std::vector<Step> steps;
    int num_slots{0};
    int out_slot{-1};
};

/// MLIR Emitter - Creates MLIR module using OpBuilder from computational graph
class MLIREmitter {
public:
    MLIREmitter();
    ~MLIREmitter();

    /// Emit MLIR module from a compilation plan
    /// Returns: MLIR module and serialized string representation
    std::pair<mlir::OwningOpRef<mlir::ModuleOp>, std::string> 
    emitModule(const Plan& plan);

    /// Get the MLIR context
    std::shared_ptr<mlir::MLIRContext> getContext() { return context_; }

private:
    std::shared_ptr<mlir::MLIRContext> context_;

    /// Convert ag::Dtype to MLIR Type
    mlir::Type dtypeToMLIRType(mlir::OpBuilder& builder, OwnTensor::Dtype dtype);

    /// Create MLIR tensor type from shape and dtype
    mlir::RankedTensorType createTensorType(
        mlir::OpBuilder& builder,
        const std::vector<int64_t>& shape,
        OwnTensor::Dtype dtype
    );

    /// Register required dialects
    void registerDialects();
};

} // namespace jit
} // namespace ag
