// =====================
// file: cgadimpl/include/ag/graph.hpp (declarations only)
// =====================
#pragma once
#include <memory>
#include <vector>
#include "tensor.hpp"
#include "ad/schema.hpp"

namespace ag {
struct Node;
struct Value {
    std::shared_ptr<Node> node;
    Value();    
    explicit Value(std::shared_ptr<Node> n);
    const std::vector<int64_t>& shape() const;
    std::pair<int, int> shape_2d() const;
    Tensor& val();
    const Tensor& val() const;
    Tensor& grad();
    const Tensor& grad() const;
};

struct Node : std::enable_shared_from_this<Node> {
    Tensor value;
    Tensor grad;    
    std::vector<std::shared_ptr<Node>> inputs;
    std::vector<Value> saved_inputs;
    std::vector<std::shared_ptr<Tensor>> tape;
    std::vector<uint8_t> saved_rng_blob;
    const char* debug_name{""};
    Op op{Op::Leaf};
    bool requires_grad_flag_{false};
    bool is_checkpoint{false};
    bool has_saved_rng{false};
    bool requires_grad() const { return requires_grad_flag_; }
    const std::vector<int64_t>& shape() const { return value.shape().dims; }
    Node(const Tensor& v, Op op_, bool req_grad, const char* nm="");
    Node() = default;
};

inline Value make_tensor(const Tensor& v, const char* name = "") {
    return Value(std::make_shared<Node>(v, Op::Leaf, v.requires_grad(), name));
}

std::vector<Node*> topo_from(Node* root);
    
// ---- Lightweight trace→compile→replay (CPU) ----
namespace jit {

struct CompileOptions {
    bool use_cuda_graph = false; // ignored for now (no CUDA)
};

struct Compiled {
    const std::string& getMLIRSource() const;
    
    // Get the in-memory MLIR module (can be null if emission failed)
    // Returns mlir::ModuleOp* - cast to use
    void* getMLIRModule() const;
    
    // Opaque impl; created by compile()
    struct Impl;
    std::shared_ptr<Impl> p;
    std::string mlir_source;  // String-based MLIR (legacy/fallback)
    std::string mlir_module_str;  // MLIR from OpBuilder emission (serialized)
    
    // In-memory MLIR module (use this for passes/lowering)
    // Stored as void* to avoid MLIR header dependencies in this header
    std::shared_ptr<void> mlir_module;

    // Run with external inputs/params. Returns false if shape guard fails.
    bool run(const std::vector<Tensor*>& inputs,
             const std::vector<Tensor*>& params,
             Tensor& out) const;
};

// Build a compiled plan from a finished forward Value (dynamic graph).
// 'inputs' and 'params' enumerate leaf Values whose storage is provided at run().
Compiled compile(const Value& output,
                 const std::vector<Value>& inputs,
                 const std::vector<Value>& params,
                 const CompileOptions& opts = {});

} // namespace jit

} // namespace ag
