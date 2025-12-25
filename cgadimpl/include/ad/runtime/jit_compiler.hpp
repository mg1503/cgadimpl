#pragma once
#include "ad/core/graph.hpp"
#include "ad/runtime/runtime.hpp"
#include "ad/core/mlir_emitter.hpp"
#include <vector>
#include <variant>
#include <memory>
#include <string>

namespace ag::jit {

// ===================================================================
// JIT Compiler Interface
// ===================================================================

struct Compiled {
    struct Impl;
    std::shared_ptr<Impl> p;

    // MLIR data
    std::string mlir_source;
    std::shared_ptr<void> mlir_module;
    std::string mlir_module_str;
    
    // Execute the compiled plan
    bool run(const std::vector<Tensor*>& inputs,
             const std::vector<Tensor*>& params,
             Tensor& out) const;

    const std::string& getMLIRSource() const;
    void* getMLIRModule() const;
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
