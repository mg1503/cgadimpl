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
#include "ad/autodiff/checkpoint.hpp"
#include "ad/core/ReadyQueue.hpp" // custom queue for our dependency counter algorithm to be implemented in the .backward() fucntion.
namespace ag {

void zero_grad(const Value& root){
    auto order = topo_from(root.node.get());
    for (Node* n : order) {
        if (n->requires_grad()) {
            Dtype grad_dtype = n->value.dtype();
            if (is_float(grad_dtype)) {
                grad_dtype = Dtype::Float32;
            }
            n->grad = Tensor::zeros(n->value.shape(), TensorOptions().with_dtype(grad_dtype).with_device(n->value.device()));
        }
    }
}

#pragma omp parallel for

void backward(const Value& root, const Tensor* grad_seed, bool enable_parallel){
    auto order = topo_from(root.node.get());

    // Initialize dependency counters
    for (Node* n : order) {
        n->child_grad_count = 0;
    }

    // Count how many children will send gradients to each parent
    for (Node* n : order) {
        for (auto& parent : n->inputs){
            if (parent && parent->requires_grad()) {
                parent->child_grad_count++;
            }
        }
    }

    // Seed the root gradient
    if (root.node->requires_grad()){
        if(grad_seed){
            root.node->grad = *grad_seed;
        }else{
            Dtype grad_dtype = root.node->value.dtype();
            if (is_float(grad_dtype)) {
                grad_dtype = Dtype::Float32;
            }
            auto opts = TensorOptions().with_dtype(grad_dtype).with_device(root.node->value.device());
            if (root.node->value.numel() == 1) {
                root.node->grad = OwnTensor::Tensor::ones(Shape{{1}}, opts);
            } else {
                root.node->grad = OwnTensor::Tensor::ones(root.node->value.shape(), opts);
            }
        }
    }

    // Dependency tracking for correctness (always needed)
    // Count how many nodes will be processed
    int num_compute_nodes = 0;
    for (Node* n : order) {
        if (n->requires_grad()) {
            num_compute_nodes++;
        }
    }

    // User chose sequential execution
    if (!enable_parallel) {
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            Node* n = *it;
            if (!n->requires_grad()) continue;
            
            const Tensor& gy = n->grad;
            
            if (n->is_checkpoint && n->value.numel() == 0) {
                if (!ag::checkpoint_impl::recompute_subgraph(n->shared_from_this())) {
                    throw std::runtime_error("autodiff: failed to recompute checkpointed node during backward");
                }
            }
            
            VjpFn fn = vjp_lookup(n->op);
            if (fn) fn(n, gy);

            for (auto& hook : n->post_acc_grad_hooks) {
                hook(n);
            }
        }
        return;
    }

    // PARALLEL EXECUTION (user opted-in via enable_parallel=true)
    std::atomic<int> pending_tasks{num_compute_nodes};

    readyqueue rq;
    // Push all nodes that are initially ready (no children waiting to send gradients)
     for (Node* n : order) {
        if (n->requires_grad() && n->child_grad_count == 0){
            // Trigger hooks for initial ready nodes (like root or independent leaves)
            for (auto& hook : n->post_acc_grad_hooks) hook(n);

            if (!n->is_leaf) {
                rq.push(n);
            } else {
                pending_tasks--;
            }
        }
    }
    
    //worker task
    auto workertask = [&](){
        while (true){
            Node* node = rq.pop();
            if ( node == nullptr ){
                break;
            }
            
            VjpFn vjpfn = vjp_lookup(node->op);

            // Execute checkpointing if needed
            if (node->is_checkpoint && node->value.numel() == 0) {
                if (!ag::checkpoint_impl::recompute_subgraph(node->shared_from_this())) {
                    pending_tasks--;
                    continue;
                }
            }
            
            // Execute VJP function
            if (vjpfn){
                vjpfn(node, node->grad);
            }

            // Update parent counters
            for (auto& parent_ptr : node->inputs){
                Node* parent = parent_ptr.get();
                if (!parent || !parent->requires_grad()) continue;

                // Decrement counter 
                if (parent->child_grad_count.fetch_sub(1) == 1){
                    for (auto& hook : parent->post_acc_grad_hooks) hook(parent);
                    rq.push(parent);
                }
            }
            pending_tasks--;
        }
    };

    //launch worker threads
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;
    for (int i = 0; i < num_threads; ++i){
        workers.emplace_back(workertask);
    }

    // Main thread waits for all tasks to complete
    while(pending_tasks > 0){
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    // Shutdown after all tasks complete
    rq.shutdown();
    for (auto& worker : workers){
        worker.join();
    }



    // // for (Node* n : order) {
        
    // //     if (n->requires_grad() /*&& n->grad.numel() == 0*/) {
    // //         n->grad = Tensor::zeros(n->value.shape(), ag::options(n->value));
    // //     }
    // // }

    //  // seed
    // if (root.node->requires_grad()) {
    //     if (grad_seed) {
    //         root.node->grad = *grad_seed;
    //     } else {
    //         // Use the new factories and get options from the value tensor
    //         auto opts = ag::options(root.node->value);
    //         if (root.node->value.numel() == 1) {
    //             root.node->grad.fill(1.0f);
    //         } else {
    //             root.node->grad = OwnTensor::Tensor::ones(root.node->value.shape(), opts);
    //         }
    //     }
    // }

    // // reverse topo
    // for (auto it = order.rbegin(); it != order.rend(); ++it) {
    //     Node* n = *it;
    //     // The requires_grad() check is now a function call
    //     if (!n->requires_grad()) continue;
    //     const Tensor& gy = n->grad;

    //     ag::debug::on_backprop_step(n, gy); // (optional) prints one line per node

    //     if (n->is_checkpoint && n->value.numel() == 0) {
    //     if (!ag::checkpoint_impl::recompute_subgraph(n->shared_from_this())) {
    //         throw std::runtime_error("autodiff: failed to recompute checkpointed node during backward");
    //     }
    //     }
        
    //     // Phase 1.1: is_leaf handling
    //     // Only compute VJP for non-leaf nodes (leaf nodes only accumulate, no backward op)
    //     if (!n->is_leaf) {
    //         //  this part calculates and accumulates gradients into parent nodes
    //         VjpFn fn = vjp_lookup(n->op);
    //         if (fn) fn(n, gy); // handler accumulates into parents
    //     }
    // }
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

        //  this part calculates and accumulates gradients
        JvpFn fn = jvp_lookup(n->op);
        if (fn) t = fn(n, tangent_of);

        T[n] = t;
    }
    return T[root.node.get()];
}

} // namespace ag
