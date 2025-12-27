// =============================================
// cgadimpl/src/core/autodiff.cpp
// Summary of the Changes
//     zero_grad():
//         Tensor::zeros(n->tensor.shape()) was changed to OwnTensor::Tensor::zeros(n->tensor.shape(), ag::options(n->tensor)).
//         Reason: This is the most important pattern. We must use the ag::options() helper to create a new zero-filled tensor that has the same device (CPU/CUDA) and dtype as the value tensor it corresponds to.
//     backward():
//         The grad_seed logic was rewritten to use OwnTensor::Tensor::ones with the correct Shape and TensorOptions, again ensuring the seed tensor is created on the correct device.
//         The check n->tensor.size() == 0 was correctly changed to n->tensor.numel() == 0 to match the new API.
//     jvp():
//         The initial return Tensor{} was changed to return Tensor{Shape{}, TensorOptions{}} to correctly default-construct an empty tensor.
//         T.reserve(order.numel()) was a bug. order is a std::vector, so we use order.size().
//         The static fallback tensor Z was changed to be a correctly default-constructed empty tensor.
//         The creation of the tangent tensor t was changed from Tensor::zeros(n->tensor) to the correct OwnTensor::Tensor::zeros(n->tensor.shape(), ag::options(n->tensor)) to ensure it's on the right device.
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
            n->tensor.set_requires_grad(true);
            n->tensor.grad_view().fill(0.0f);
        }
    }
}

#pragma omp parallel for

void backward(const Value& root, const Tensor* grad_seed, bool enable_parallel){
    std::cerr << "[DEBUG] backward() called, enable_parallel=" << enable_parallel << std::endl;
    auto order = topo_from(root.node.get());
    std::cerr << "[DEBUG] topo_from returned " << order.size() << " nodes" << std::endl;

    // Initialize dependency counters
    for (Node* n : order) {
        n->child_grad_count = 0;
    }
    std::cerr << "[DEBUG] Initialized counters" << std::endl;

    // Count how many children will send gradients to each parent
    for (Node* n : order) {
        for (auto& edge : n->next_edges){
            if (edge.function && edge.function->requires_grad()) {
                edge.function->child_grad_count++;
            }
        }
    }
    std::cerr << "[DEBUG] Counted dependencies" << std::endl;

    // Seed the root gradient
    if (root.node->requires_grad()){
        root.node->tensor.set_requires_grad(true);
        Tensor g = root.node->tensor.grad_view();
        if(grad_seed){
            g.fill(0.0f);
            g += *grad_seed; // Accumulate seed into gradient
        }else{
            auto opts = ag::options(root.node->tensor);
            g.fill(1.0f);
        }
    }

    // Dependency tracking for correctness (always needed)
    // Count only NON-LEAF nodes that will be processed by workers
    // Leaf nodes are handled separately when they become ready (see line ~150)
    int num_compute_nodes = 0;
    for (Node* n : order) {
        if (n->requires_grad() && !n->is_leaf()) {
            num_compute_nodes++;
        }
    }

    // User chose sequential execution
    if (!enable_parallel) {
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            Node* n = *it;
            if (!n->requires_grad()) continue;
            
            Tensor gy = n->tensor.grad_view();
            
            if (n->is_checkpoint && n->tensor.numel() == 0) {
                if (!ag::checkpoint_impl::recompute_subgraph(n->shared_from_this())) {
                    throw std::runtime_error("autodiff: failed to recompute checkpointed node during backward");
                }
            }
            
            VjpFn fn = vjp_lookup(n->op);
            if (fn) fn(n, gy);
        }
        return;
    }

    // PARALLEL EXECUTION (user opted-in via enable_parallel=true)
    std::atomic<int> pending_tasks{num_compute_nodes};

    std::cerr << "[DEBUG] Starting parallel backward: pending_tasks=" << pending_tasks << ", non-leaf nodes=" << num_compute_nodes << std::endl;

    readyqueue rq;
    
    // The root node always starts with child_grad_count == 0 (it's the output of forward pass)
    // Push the root first to kick off the backward pass
    int initial_ready = 0;
    if (root.node->requires_grad() && !root.node->is_leaf()) {
        rq.push(root.node.get());
        initial_ready++;
        std::cerr << "[DEBUG] Pushed root node (op=" << (int)root.node->op << ")" << std::endl;
    } else {
        std::cerr << "[DEBUG] Root is leaf or doesn't require grad" << std::endl;
    }
    
    // Push all other non-leaf nodes that are initially ready (no children waiting to send gradients)
    for (Node* n : order) {
        if (n == root.node.get()) continue; // Skip root, already added
        if (n->requires_grad() && !n->is_leaf() && n->child_grad_count == 0){
            rq.push(n);
            initial_ready++;
        }
    }
    std::cerr << "[DEBUG] Initial ready queue size: " << initial_ready << std::endl;
    
    //worker task
    std::atomic<int> nodes_processed{0};
    auto workertask = [&](){
        while (true){
            Node* node = rq.pop();
            if ( node == nullptr ){
                break;
            }
            
            nodes_processed++;
            if (nodes_processed % 5 == 1) {
                std::cerr << "[DEBUG] Processing node #" << nodes_processed << " (op=" << (int)node->op << "), pending=" << pending_tasks << std::endl;
            }
            
            VjpFn vjpfn = vjp_lookup(node->op);

            // Execute checkpointing if needed
            if (node->is_checkpoint && node->tensor.numel() == 0) {
                if (!ag::checkpoint_impl::recompute_subgraph(node->shared_from_this())) {
                    pending_tasks--;
                    continue;
                }
            }
            
            // Execute VJP function
            if (vjpfn){
                vjpfn(node, node->tensor.grad_view());
            } else {
                std::cerr << "[DEBUG] No VJP for op=" << (int)node->op << std::endl;
            }

            // Update parent counters
            int parents_made_ready = 0;
            for (auto& edge : node->next_edges){
                Node* parent = edge.function.get();
                if (!parent || !parent->requires_grad()) continue;

                // Decrement counter 
                if (parent->child_grad_count.fetch_sub(1) == 1){
                    // Parent is now ready (all children have sent gradients)
                    if (!parent->is_leaf()) {
                        // Non-leaf nodes need VJP, push to queue for processing
                        rq.push(parent);
                        parents_made_ready++;
                    }
                    // Leaf nodes are ready but don't need processing, so do nothing
                }
            }
            if (parents_made_ready > 0 && nodes_processed % 5 == 1) {
                std::cerr << "[DEBUG] Made " << parents_made_ready << " parents ready" << std::endl;
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
    int wait_iterations = 0;
    while(pending_tasks > 0){
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_iterations++;
        if (wait_iterations % 10 == 0) {
            std::cerr << "[DEBUG] Still waiting... pending=" << pending_tasks << ", processed=" << nodes_processed << std::endl;
        }
        if (wait_iterations > 100) {
            std::cerr << "[ERROR] Timeout after 10 seconds! pending=" << pending_tasks << ", processed=" << nodes_processed << std::endl;
            break;
        }
    }
    
    // Shutdown after all tasks complete
    rq.shutdown();
    for (auto& worker : workers){
        worker.join();
    }



    // // for (Node* n : order) {
        
    // //     if (n->requires_grad() /*&& n->tensor.grad_view().numel() == 0*/) {
    // //         n->tensor.grad_view() = Tensor::zeros(n->tensor.shape(), ag::options(n->tensor));
    // //     }
    // // }

    //  // seed
    // if (root.node->requires_grad()) {
    //     if (grad_seed) {
    //         root.node->tensor.grad_view() = *grad_seed;
    //     } else {
    //         // Use the new factories and get options from the value tensor
    //         auto opts = ag::options(root.node->tensor);
    //         if (root.node->tensor.numel() == 1) {
    //             root.node->tensor.grad_view().fill(1.0f);
    //         } else {
    //             root.node->tensor.grad_view() = OwnTensor::Tensor::ones(root.node->tensor.shape(), opts);
    //         }
    //     }
    // }

    // // reverse topo
    // for (auto it = order.rbegin(); it != order.rend(); ++it) {
    //     Node* n = *it;
    //     // The requires_grad() check is now a function call
    //     if (!n->requires_grad()) continue;
    //     const Tensor& gy = n->tensor.grad_view();

    //     ag::debug::on_backprop_step(n, gy); // (optional) prints one line per node

    //     if (n->is_checkpoint && n->tensor.numel() == 0) {
    //     if (!ag::checkpoint_impl::recompute_subgraph(n->shared_from_this())) {
    //         throw std::runtime_error("autodiff: failed to recompute checkpointed node during backward");
    //     }
    //     }
        
    //     // Phase 1.1: is_leaf handling
    //     // Only compute VJP for non-leaf nodes (leaf nodes only accumulate, no backward op)
    //     if (!n->is_leaf()) {
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
        Tensor t = OwnTensor::Tensor::zeros(n->tensor.shape(), ag::options(n->tensor));
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
