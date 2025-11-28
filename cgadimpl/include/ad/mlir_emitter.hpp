// =====================
// file: cgadimpl/include/ad/mlir_emitter.hpp
// =====================
#pragma once
#include <string>
#include <vector>
#include "tensor.hpp"
#include "ad/schema.hpp"

namespace ag::jit {

// Forward declarations
struct Plan;
struct Step;
struct Arg;

namespace mlir_emit {

// Convert a JIT Plan to MLIR Nova dialect text representation
std::string emitNovaDialect(const Plan& plan, const std::string& function_name = "jit_function");

// Helper: Convert tensor dtype to MLIR type string (e.g., "f32", "i32")
std::string dtypeToMLIRType(OwnTensor::Dtype dtype);

// Helper: Convert tensor shape to MLIR tensor type (e.g., "tensor<4x8xf32>")
std::string shapeToMLIRType(const std::vector<int64_t>& shape, OwnTensor::Dtype dtype);

// Helper: Map JIT Op enum to Nova operation name
std::string opToNovaOp(Op op);

// Helper: Check if operation is a reduction that needs special handling
bool isReductionOp(Op op);

// Helper: Get reduction kind string for Nova reduce operation
std::string getReductionKind(Op op);

// Helper: Check if operation needs dimension attribute (e.g., RowSum, RowMax)
bool needsDimensionAttr(Op op);

} // namespace mlir_emit
} // namespace ag::jit
