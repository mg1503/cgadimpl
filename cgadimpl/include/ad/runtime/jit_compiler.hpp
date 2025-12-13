#pragma once
#include "ad/core/graph.hpp"
#include "ad/runtime/runtime.hpp"
#include <vector>
#include <variant>
#include <memory>
// #include "TensorLib.h" // Assuming needed for some types, or just forward declare if possible

namespace ag::jit {

// Forward declarations
struct Signature;
struct Step;
struct Plan;

// ===================================================================
// JIT Compiler Interface
// ===================================================================

struct Compiled {
    struct Impl;
    std::shared_ptr<Impl> p;
    
    // Execute the compiled plan
    bool run(const std::vector<Tensor*>& inputs,
             const std::vector<Tensor*>& params,
             Tensor& out) const;
};

struct CompileOptions {
    bool use_cuda_graph = false;
    // ... optimization flags ...
};

Compiled compile(const Value& output,
                 const std::vector<Value>& inputs,
                 const std::vector<Value>& params,
                 const CompileOptions& opts = {});

} // namespace ag::jit
