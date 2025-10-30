// =====================
// file: cgadimpl/include/ag/graph.hpp (declarations only)
// =====================
#pragma once
#include <memory>
#include <vector>
#include "tensor.hpp"
#include "ad/schema.hpp"

namespace ag {

struct Node; // Forward declaration

struct Value {
    std::shared_ptr<Node> node;

    Value();    
    explicit Value(std::shared_ptr<Node> n);

    // Helper for backward compatibility with code expecting a 2D shape pair.
    std::pair<int, int> shape() const;
};

struct Node : std::enable_shared_from_this<Node> {
    Op op{Op::Leaf};
    std::vector<std::shared_ptr<Node>> inputs;

    Tensor value; // This is an OwnTensor::Tensor
    Tensor grad;  // This is also an OwnTensor::Tensor

    bool is_checkpoint{false};
    std::vector<Value> saved_inputs;
    
    // The `requires_grad` property now lives on the tensor itself.
    bool requires_grad() const { return value.requires_grad(); }

    const char* debug_name{""};
    std::vector<std::shared_ptr<Tensor>> tape;

    // A single, clean constructor.
    Node(const Tensor& v, Op op_, const char* nm="");
    Node() = default;
};


// The single, unified factory function.
Value make_tensor(const Tensor& v, const char* name = "");

// Convenience wrappers that use the new TensorOptions API.
inline Value constant(const Tensor& v, const char* name="const") {
    return make_tensor(v, name);
}

inline Value param(const Tensor& v, const char* name="param") {
    // Create a new tensor from the same data but with requires_grad=true
    Tensor param_tensor(v.shape(), v.dtype(), v.device(), /*requires_grad=*/true);
    param_tensor.copy_(v); // Copy the data
    return make_tensor(param_tensor, name);
}

std::vector<Node*> topo_from(Node* root);


    
// ---- Lightweight trace→compile→replay (CPU) ----
// namespace jit {

// struct CompileOptions {
//     bool use_cuda_graph = false; // ignored for now (no CUDA)
// };

// struct Compiled {
//     // Opaque impl; created by compile()
//     struct Impl;
//     std::shared_ptr<Impl> p;

//     // Run with external inputs/params. Returns false if shape guard fails.
//     bool run(const std::vector<Tensor*>& inputs,
//              const std::vector<Tensor*>& params,
//              Tensor& out) const;
// };

// // Build a compiled plan from a finished forward Value (dynamic graph).
// // 'inputs' and 'params' enumerate leaf Values whose storage is provided at run().
// Compiled compile(const Value& output,
//                  const std::vector<Value>& inputs,
//                  const std::vector<Value>& params,
//                  const CompileOptions& opts = {});

// } // namespace jit

} // namespace ag
