// =====================
// file: cgadimpl/src/graph.cpp
// =====================
#include "ad/core/graph.hpp"
#include <unordered_set>
#include <functional>
#include <cassert>
#include <sstream>
#include <iostream> // Added for printing


namespace ag {

// --- Node Implementation ---
// Node::Node() = default; 
Node::Node(const Tensor& v, Op op_, bool req_grad, const char* nm) 
    : tensor(v), 
      op(op_), 
      debug_name(nm),
      is_leaf(op_ == Op::Leaf)  // Phase 1.1: Mark leaf nodes
{
    // Phase 1.3: Capture execution context
    creation_context.stream = current_stream();
    creation_context.device = v.device();
    
   if (req_grad && !tensor.requires_grad()) {
        tensor.set_requires_grad(true);
    }
}

// --- Value Implementation ---
// ADDED: Implement the Value helper functions
Tensor& Value::val() { return node->tensor; }
const Tensor& Value::val() const { return node->tensor; }
Tensor Value::grad() { return node->tensor.grad_view(); }
Tensor Value::grad() const { return node->tensor.grad_view(); }
Value::Value() = default;
Value::Value(std::shared_ptr<Node> n) : node(std::move(n)) {}

// NEW: Implementation for the real shape()
const std::vector<int64_t>& Value::shape() const {
    return node->tensor.shape().dims;
}
// 2d helper
std::pair<int, int> Value::shape_2d() const {
    const auto& dims = node->tensor.shape().dims;
    if (dims.size() == 0) return {0, 0};
    if (dims.size() == 1) return {1, static_cast<int>(dims[0])};
    // For 2D or more, return the first two dimensions.
    return {static_cast<int>(dims[0]), static_cast<int>(dims[1])};
}

// // --- Factory Implementation ---
// Value make_tensor(const Tensor& v, const char* name) {
//     return Value(std::make_shared<Node>(v, Op::Leaf, name));
// }

// --- Internal implementation for graph traversal ---
static std::vector<Node*> build_topo_order_impl(Node* root) {
    std::vector<Node*> order; order.reserve(256);
    std::unordered_set<Node*> vis; vis.reserve(256);
    std::function<void(Node*)> dfs = [&](Node* n){ if(!n || vis.count(n)) return; vis.insert(n); for(auto& p : n->inputs) dfs(p.get()); order.push_back(n); };
    dfs(root);
    return order; // parents before child
}

// --- Graph Traversal ---
std::vector<Node*> topo_from(Node* root){
    return build_topo_order_impl(root);
}

} // namespace ag