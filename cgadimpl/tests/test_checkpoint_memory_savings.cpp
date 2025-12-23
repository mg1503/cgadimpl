#include "ad/graph.hpp"
#include "ad/autodiff.hpp"
#include "ad/checkpoint.hpp"
#include "ad/careful_deletion.hpp"
#include "tensor.hpp"
#include <iostream>
#include <vector>
#include <numeric>

using namespace ag;

// ============================================================================
// Helper: Calculate Total Graph Memory
// ============================================================================

size_t calculate_graph_memory(const Value& root) {
    if (!root.node) return 0;
    
    auto nodes = topo_from(root.node.get());
    size_t total_bytes = 0;
    
    for (Node* n : nodes) {
        // Count value memory
        if (n->value.numel() > 0) {
            total_bytes += n->value.numel() * sizeof(float);
        }
        // Count gradient memory
        if (n->grad.numel() > 0) {
            total_bytes += n->grad.numel() * sizeof(float);
        }
    }
    
    return total_bytes;
}

// ============================================================================
// Model Definition
// ============================================================================

struct DeepModel {
    std::vector<Value> weights;
    std::vector<Value> biases;
    int depth;
    int hidden_dim;
    
    DeepModel(int d, int h) : depth(d), hidden_dim(h) {
        auto opts = TensorOptions().with_req_grad(true);
        for (int i = 0; i < depth; ++i) {
            weights.push_back(make_tensor(Tensor::randn(Shape{{h, h}}, opts), ("w" + std::to_string(i)).c_str()));
            biases.push_back(make_tensor(Tensor::randn(Shape{{1, h}}, opts), ("b" + std::to_string(i)).c_str()));
        }
    }
    
    Value forward(Value x, bool use_checkpointing) {
        for (int i = 0; i < depth; ++i) {
            // Linear layer
            x = matmul(x, weights[i]) + biases[i];
            x = relu(x);
            
            // Checkpoint every 3rd layer if enabled
            if (use_checkpointing && (i > 0) && (i % 3 == 0) && (i < depth - 1)) {
                checkpoint_impl::mark_node_checkpoint(x.node, CheckpointOptions());
            }
        }
        return x;
    }
    
    std::vector<Value> parameters() {
        std::vector<Value> params;
        params.insert(params.end(), weights.begin(), weights.end());
        params.insert(params.end(), biases.begin(), biases.end());
        return params;
    }
};

// ============================================================================
// Test Runner
// ============================================================================

void run_memory_savings_test() {
    std::cout << "==================================================\n";
    std::cout << "      Checkpointing Memory Savings Demo           \n";
    std::cout << "==================================================\n\n";
    
    int depth = 20;
    int hidden_dim = 512;
    int batch_size = 64;
    
    std::cout << "Model Config:\n";
    std::cout << "  - Depth: " << depth << " layers\n";
    std::cout << "  - Hidden Dim: " << hidden_dim << "\n";
    std::cout << "  - Batch Size: " << batch_size << "\n\n";
    
    DeepModel model(depth, hidden_dim);
    Value input = make_tensor(Tensor::randn(Shape{{batch_size, hidden_dim}}, TensorOptions()), "input");
    
    // ------------------------------------------------------------------------
    // Scenario 1: Standard Training (No Checkpointing)
    // ------------------------------------------------------------------------
    std::cout << "--- Scenario 1: Standard Training (No Checkpointing) ---\n";
    
    // Forward
    Value out_std = model.forward(input, false);
    
    // In standard training, we keep ALL activations for backward
    // So we don't delete anything.
    
    size_t mem_std = calculate_graph_memory(out_std);
    std::cout << "  Peak Memory (Activations): " << (mem_std / 1024.0 / 1024.0) << " MB\n";
    
    // Cleanup for next run
    out_std = Value(); // Release reference
    
    // ------------------------------------------------------------------------
    // Scenario 2: With Gradient Checkpointing
    // ------------------------------------------------------------------------
    std::cout << "\n--- Scenario 2: With Gradient Checkpointing ---\n";
    
    // Forward
    Value out_cp = model.forward(input, true);
    
    // With checkpointing, we can aggressively delete non-checkpoint nodes
    // The backward pass will recompute them as needed.
    std::cout << "  Performing memory cleanup (simulating training loop)...\n";
    
    // STRATEGY:
    // 1. Identify "Anchor Checkpoints" (the ones marked by the model).
    // 2. Mark ALL other intermediate nodes as checkpoints so autodiff recomputes them.
    // 3. Protect Anchors from deletion.
    // 4. Delete everything else.
    
    std::unordered_set<Node*> anchors;
    auto nodes = topo_from(out_cp.node.get());
    
    // Step 1: Find anchors
    for (Node* n : nodes) {
        if (n->is_checkpoint) {
            anchors.insert(n);
        }
    }
    std::cout << "  Identified " << anchors.size() << " anchor checkpoints.\n";
    
    // Step 2: Mark intermediates
    int marked_intermediates = 0;
    for (Node* n : nodes) {
        if (n->op != Op::Leaf && !n->is_checkpoint) {
            // Mark as checkpoint for recomputation
            checkpoint_impl::mark_node_checkpoint(n->shared_from_this(), CheckpointOptions());
            marked_intermediates++;
        }
    }
    std::cout << "  Marked " << marked_intermediates << " intermediates for recomputation.\n";
    
    // Step 3 & 4: Sweep with protection
    memory::sweep_safe_nodes(out_cp, memory::DeletePolicy::ForwardPass, anchors);
    
    size_t mem_cp = calculate_graph_memory(out_cp);
    std::cout << "  Peak Memory (After Cleanup): " << (mem_cp / 1024.0 / 1024.0) << " MB\n";
    
    // ------------------------------------------------------------------------
    // Results
    // ------------------------------------------------------------------------
    std::cout << "\n--------------------------------------------------\n";
    std::cout << "Results Summary:\n";
    std::cout << "  Standard Memory:     " << (mem_std / 1024.0 / 1024.0) << " MB\n";
    std::cout << "  Checkpointed Memory: " << (mem_cp / 1024.0 / 1024.0) << " MB\n";
    
    size_t saved = mem_std - mem_cp;
    double percent = 100.0 * saved / mem_std;
    
    std::cout << "  Memory Saved:        " << (saved / 1024.0 / 1024.0) << " MB (" << percent << "%)\n";
    std::cout << "--------------------------------------------------\n";
    
    if (saved > 0) {
        std::cout << "\n✅ SUCCESS: Checkpointing successfully reduced memory usage!\n";
    } else {
        std::cout << "\n❌ FAILURE: No memory savings observed.\n";
    }
    
    // Verify Backward still works
    std::cout << "\nVerifying backward pass works with checkpoints...\n";
    try {
        Value loss = sum(out_cp);
        backward(loss);
        std::cout << "  Backward pass completed successfully.\n";
    } catch (const std::exception& e) {
        std::cout << "  ❌ Backward pass failed: " << e.what() << "\n";
    }
}

int main() {
    try {
        run_memory_savings_test();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
