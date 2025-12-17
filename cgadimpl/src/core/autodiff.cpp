// =============================================
// cgadimpl/src/core/autodiff.cpp
// Summary of the Changes
//     zero_grad():
//         Tensor::zeros(n->value.shape()) was changed to OwnTensor::Tensor::zeros(n->value.shape(), ag::options(n->value)).
//         Reason: This is the most important pattern. We must use the ag::options() helper to create a new zero-filled tensor that has the same device (CPU/CUDA) and dtype as the value tensor it corresponds to.
//     backward():
//         The grad_seed logic was rewritten to use OwnTensor::Tensor::ones with the correct Shape and TensorOptions, again ensuring the seed tensor is created on the correct device.
//         The check n->value.size() == 0 was correctly changed to n->value.numel() == 0 to match the new API.
//     jvp():
//         The initial return Tensor{} was changed to return Tensor{Shape{}, TensorOptions{}} to correctly default-construct an empty tensor.
//         T.reserve(order.numel()) was a bug. order is a std::vector, so we use order.size().
//         The static fallback tensor Z was changed to be a correctly default-constructed empty tensor.
//         The creation of the tangent tensor t was changed from Tensor::zeros(n->value) to the correct OwnTensor::Tensor::zeros(n->value.shape(), ag::options(n->value)) to ensure it's on the right device.
// You have now updated all the core logic of the autodiff engine to be fully compatible with the new OwnTensor library. This was a critical step.
// =============================================
#include <unordered_map>
#include <stdexcept>
#include "ad/autodiff/autodiff.hpp"
#include "ad/detail/autodiff_ops.hpp"
#include "ad/utils/debug.hpp"
#include <ad/autodiff/checkpoint.hpp>
#include <ad/core/ReadyQueue.hpp> // custom queue for our dependency counter algorithm to be implemented in the .backward() fucntion.
namespace ag {

void zero_grad(const Value& root){
    auto order = topo_from(root.node.get());
    for (Node* n : order) if (n->requires_grad()) n->grad = Tensor::zeros(n->value.shape(), ag::options(n->value));
}

void backward(const Value& root, const Tensor* grad_seed){
    auto order = topo_from(root.node.get());

    // for (Node* n : order) {
        
    //     if (n->requires_grad() /*&& n->grad.numel() == 0*/) {
    //         n->grad = Tensor::zeros(n->value.shape(), ag::options(n->value));
    //     }
    // }

     // seed
    if (root.node->requires_grad()) {
        if (grad_seed) {
            root.node->grad = *grad_seed;
        } else {
            // Use the new factories and get options from the value tensor
            auto opts = ag::options(root.node->value);
            if (root.node->value.numel() == 1) {
                root.node->grad.fill(1.0f);
            } else {
                root.node->grad = OwnTensor::Tensor::ones(root.node->value.shape(), opts);
            }
        }
    }

    // reverse topo
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node* n = *it;
        // The requires_grad() check is now a function call
        if (!n->requires_grad()) continue;
        const Tensor& gy = n->grad;

        ag::debug::on_backprop_step(n, gy); // (optional) prints one line per node

        if (n->is_checkpoint && n->value.numel() == 0) {
        if (!ag::checkpoint_impl::recompute_subgraph(n->shared_from_this())) {
            throw std::runtime_error("autodiff: failed to recompute checkpointed node during backward");
        }
        }
        
        // Phase 1.1: is_leaf handling
        // Only compute VJP for non-leaf nodes (leaf nodes only accumulate, no backward op)
        if (!n->is_leaf) {
            //  this part calculates and accumulates gradients into parent nodes
            VjpFn fn = vjp_lookup(n->op);
            if (fn) fn(n, gy); // handler accumulates into parents
        }
    }
}

Tensor jvp(const Value& root, const std::unordered_map<Node*, Tensor>& seed){
    if (!root.node) return Tensor{Shape{}, TensorOptions{}}; // Return a valid empty tensor
    
    auto order = topo_from(root.node.get());
    std::unordered_map<Node*, Tensor> T;
    T.reserve(order.size()); // Use vector's .size(), not a tensor's .numel()

    auto tangent_of = [&](Node* p) -> const Tensor& {
        auto it = T.find(p);
        if (it != T.end()) return it->second;
        // A static empty tensor is a good fallback
        static Tensor Z{Shape{}, TensorOptions{}}; 
        return Z;
    };

    for (Node* n : order) {
        // seed tangent for this node (if provided), else zeros of the correct shape/device
        Tensor t = OwnTensor::Tensor::zeros(n->value.shape(), ag::options(n->value));
        if (auto it = seed.find(n); it != seed.end()) {
            t = it->second;
        }

        ag::debug::on_jvp_step(n);

        JvpFn fn = jvp_lookup(n->op);
        if (fn) t = fn(n, tangent_of);

        T[n] = t;
    }
    return T[root.node.get()];
}

} // namespace ag
