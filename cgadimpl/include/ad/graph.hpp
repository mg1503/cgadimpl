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

// Value remains a lightweight user handle to a Node.
struct Value {
    std::shared_ptr<Node> node;

    Value();    
    explicit Value(std::shared_ptr<Node> n);

    // NEW: This function returns the REAL shape.
    const std::vector<int64_t>& shape() const;

    // RENAMED: This is a helper for simple 2D cases.
    std::pair<int, int> shape_2d() const;

    // We can keep these useful helpers, but they will just forward to the node.
    const Tensor& val() const;
    Tensor& grad();
    const Tensor& grad() const;
};

// Node is the core data structure with all checkpointing members restored.
struct Node : std::enable_shared_from_this<Node> {
    // --- Group 1: Largest Objects (Vectors and Tensors) ---
    // These are all large and have an 8-byte alignment requirement.
    // Their internal order doesn't matter much relative to each other.
    Tensor value{OwnTensor::Shape{}, OwnTensor::Dtype::Float32};
    Tensor grad{OwnTensor::Shape{}, OwnTensor::Dtype::Float32};

    std::vector<std::shared_ptr<Node>> inputs;
    std::vector<Value> saved_inputs;
    std::vector<std::shared_ptr<Tensor>> tape;
    std::vector<uint8_t> saved_rng_blob;
    
    // --- Group 2: 8-Byte Pointers ---
    const char* debug_name{""};

    // --- Group 3: Smallest Types (bytes and bools) ---
    // Grouping these together allows the compiler to pack them into a single 8-byte word.
    Op op{Op::Leaf};             // 1 byte
    bool is_checkpoint{false};   // 1 byte
    bool has_saved_rng{false};   // 1 byte
    // 5 bytes of padding will likely be added here by the compiler to align the whole struct.

    // --- Member Functions (no size impact) ---
    bool requires_grad() const { return value.requires_grad(); }
    const std::vector<int64_t>& shape() const { return value.shape().dims; }
    
    Node(const Tensor& v, Op op_, const char* nm="");
    Node() = default;
};


// === THE ONLY FACTORY FUNCTION NEEDED ===
// Creates a leaf node in the graph from a pre-made tensor.
// The tensor itself determines if it's a parameter or constant.
inline Value make_tensor(const Tensor& v, const char* name = "") {
    return Value(std::make_shared<Node>(v, Op::Leaf, name));
}
// ==========================================


std::vector<Node*> topo_from(Node* root);
    
    
// ---- Lightweight trace→compile→replay (CPU) ----
namespace jit {

struct CompileOptions {
    bool use_cuda_graph = false; // ignored for now (no CUDA)
};

struct Compiled {
    // Opaque impl; created by compile()
    struct Impl;
    std::shared_ptr<Impl> p;

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
