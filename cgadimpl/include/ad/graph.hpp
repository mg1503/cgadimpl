// =====================
// file: include/ag/graph.hpp (declarations only)
// =====================
#pragma once
#include <memory>
#include <vector>
#include "tensor.hpp"
#include "ad/schema.hpp"


namespace ag {

struct Node;
struct Value;

struct Node : std::enable_shared_from_this<Node>{
Op op{Op::Leaf};
std::vector<std::shared_ptr<Node>> inputs; // parents held via shared_ptr to keep them alive // parents in the compute graph
Tensor value; // forward value
Tensor grad; // same shape as value
bool requires_grad{false};
bool is_checkpoint{false};

std::vector<Value> saved_inputs;
std::vector<uint8_t> saved_rng_blob;

bool has_saved_rng{false};
const char* debug_name{""};
std::vector<std::shared_ptr<Tensor>> tape;// optional: for ops that need to save intermediates for backward

Node();
Node(const Tensor& v, bool rg, Op op_, const char* nm="");
};


struct Value { // user handle
std::shared_ptr<Node> node;
Value();    
explicit Value(std::shared_ptr<Node> n);


const Tensor& val() const;
Tensor& grad();
std::pair<int,int> shape() const;
};


Value constant(const Tensor& v, const char* name="const");
Value param (const Tensor& v, const char* name="param");


// Topological order from root (parents before child)
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
