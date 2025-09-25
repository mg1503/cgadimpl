// =====================
// file: src/graph.cpp
// =====================
#include <unordered_set>
#include <functional>
#include "ad/graph.hpp"


namespace ag {


    Node::Node() = default;
    Node::Node(const Tensor& v, bool rg, Op op_, const char* nm) : op(op_), value(v), grad(Tensor::zeros_like(v)), requires_grad(rg), debug_name(nm) {}


    Value::Value() = default;

    Value::Value(std::shared_ptr<Node> n): node(std::move(n)) {}

    const Tensor& Value::val() const { 
        return node->value; 
    }

    Tensor& Value::grad() { 
        return node->grad; 
    }
    
    std::pair<int,int> Value::shape() const { 
        return node->value.shape(); 
    }


    Value constant(const Tensor& v, const char* name){ 
        return Value(std::make_shared<Node>(v,false,Op::Leaf,name)); 
    }

    Value param (const Tensor& v, const char* name){ 
        return Value(std::make_shared<Node>(v,true ,Op::Leaf,name));
    }

    std::vector<Node*> topo_from(Node* root){
        std::vector<Node*> order; order.reserve(256);
        std::unordered_set<Node*> vis; vis.reserve(256);
        std::function<void(Node*)> dfs = [&](Node* n){ if(!n || vis.count(n)) return; vis.insert(n); for(auto& p : n->inputs) dfs(p.get()); order.push_back(n); };
        dfs(root);
        return order; // parents before child
    }


} // namespace ag