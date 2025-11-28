// =====================
// file: cgadimpl/src/core/mlir_emitter.cpp
// =====================
#include "ad/mlir_emitter.hpp"
#include <sstream>
#include <cassert>

// NOTE: The actual emitNovaDialect implementation is in graph.cpp
// where the Plan struct is fully defined. This file only contains
// the helper functions that don't need Plan access.

namespace ag::jit {
namespace mlir_emit {

std::string dtypeToMLIRType(OwnTensor::Dtype dtype) {
    switch (dtype) {
        case OwnTensor::Dtype::Float32: return "f32";
        case OwnTensor::Dtype::Float64: return "f64";
        case OwnTensor::Dtype::Int32:   return "i32";
        case OwnTensor::Dtype::Int64:   return "i64";
        // case OwnTensor::Dtype::Int8:    return "i8";
        // case OwnTensor::Dtype::Bool:    return "i1";
        default: return "f32"; // fallback
    }
}

std::string shapeToMLIRType(const std::vector<int64_t>& shape, OwnTensor::Dtype dtype) {
    std::ostringstream oss;
    oss << "tensor<";
    
    if (shape.empty() || (shape.size() == 1 && shape[0] == 1)) {
        // Scalar tensor
        oss << dtypeToMLIRType(dtype);
    } else {
        // Multi-dimensional tensor
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) oss << "x";
            oss << shape[i];
        }
        oss << "x" << dtypeToMLIRType(dtype);
    }
    
    oss << ">";
    return oss.str();
}

std::string opToNovaOp(Op op) {
    switch (op) {
        case Op::Add:       return "nova.add";
        case Op::Sub:       return "nova.sub";
        case Op::Mul:       return "nova.mul";
        case Op::MatMul:    return "nova.matmul";
        case Op::Relu:      return "nova.relu";
        case Op::Exp:       return "nova.exp";
        case Op::Log:       return "nova.log";
        case Op::Tanh:      return "nova.tanh";
        case Op::Transpose: return "nova.transpose";
        // Reductions are handled separately
        case Op::Sum:
        case Op::RowSum:
        case Op::RowMax:
        case Op::MeanAll:   return "nova.reduce";
        default:            return "nova.unknown";
    }
}

bool isReductionOp(Op op) {
    return op == Op::Sum || op == Op::RowSum || 
           op == Op::RowMax || op == Op::MeanAll;
}

std::string getReductionKind(Op op) {
    switch (op) {
        case Op::Sum:
        case Op::RowSum:    return "sum";
        case Op::RowMax:    return "max";
        case Op::MeanAll:   return "mean";
        default:            return "sum";
    }
}

bool needsDimensionAttr(Op op) {
    return op == Op::RowSum || op == Op::RowMax;
}

} // namespace mlir_emit
} // namespace ag::jit
