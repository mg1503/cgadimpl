// =====================
// file: include/ag/graph.hpp (declarations only)
// =====================
#pragma once
#include <memory>
#include <vector>
#include "ad/tensor.hpp"
#include "ad/schema.hpp"


namespace ag {


struct Node {
Op op{Op::Leaf};
std::vector<std::shared_ptr<Node>> inputs; // parents held via shared_ptr to keep them alive // parents in the compute graph
Tensor value; // forward value
Tensor grad; // same shape as value
bool requires_grad{false};
const char* debug_name{""};


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


} // namespace ag