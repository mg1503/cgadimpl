// src/autodiff.cpp
#include <unordered_map>
#include <stdexcept>
#include "ad/autodiff.hpp"
#include "ad/detail/autodiff_ops.hpp"
#include "ad/debug.hpp"
#include <ad/checkpoint.hpp>
namespace ag {

void zero_grad(const Value& root){
    auto order = topo_from(root.node.get());
    for (Node* n : order) if (n->requires_grad) n->grad = Tensor::zeros_like(n->value);
}

void backward(const Value& root, const Tensor* grad_seed){
    auto order = topo_from(root.node.get());

    // seed
    if (root.node->requires_grad) {
        root.node->grad = grad_seed ? *grad_seed
                                    : (root.node->value.size()==1 ? Tensor::ones(1,1)
                                                                  : Tensor::ones_like(root.node->value));
    }

    // reverse topo
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node* n = *it;
        if (!n->requires_grad) continue;
        const Tensor& gy = n->grad;

        ag::debug::on_backprop_step(n, gy); // (optional) prints one line per node

        if (n->is_checkpoint && n->value.size() == 0) {
        if (!ag::checkpoint_impl::recompute_subgraph(n->shared_from_this())) {
            throw std::runtime_error("autodiff: failed to recompute checkpointed node during backward");
        }
        }
        VjpFn fn = vjp_lookup(n->op);
        if (fn) fn(n, gy); // handler accumulates into parents
    }
}

Tensor jvp(const Value& root, const std::unordered_map<Node*, Tensor>& seed){
    auto order = topo_from(root.node.get());
    std::unordered_map<Node*, Tensor> T;
    T.reserve(order.size());

    auto tangent_of = [&](Node* p) -> const Tensor& {
        auto it = T.find(p);
        if (it != T.end()) return it->second;
        static Tensor Z; // fallback; shouldn't be used for leaves without seeds
        return Z;
    };

    for (Node* n : order) {
        // seed tangent for this node (if provided), else zeros
        Tensor t = Tensor::zeros_like(n->value);
        if (auto it = seed.find(n); it != seed.end()) t = it->second;

        ag::debug::on_jvp_step(n); // (optional) prints forward-mode step

        JvpFn fn = jvp_lookup(n->op);
        if (fn) t = fn(n, tangent_of);

        T[n] = t;
    }
    return T[root.node.get()];
}

} // namespace ag
