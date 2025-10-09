// // ============================================================
// // File: test_checkpoint.cpp
// // Purpose: Verify gradient checkpointing and recomputation
// // ============================================================

// #include <iostream>
// #include <vector>
// #include "ad/ag_all.hpp"

// using namespace ag;

// int main() {
//     std::cout << "===== Gradient Checkpointing Test =====\n";

//     // 1. Create some simple input tensors
//     Tensor x_data = Tensor::randn(2, 2, 42);  // small deterministic input
//     Tensor W_data = Tensor::randn(2, 2, 123);
//     Tensor b_data = Tensor::randn(2, 2, 7);

//     // 2. Wrap them as Values for the computational graph
//     Value x = constant(x_data, "x");
//     Value W = param(W_data, "W");
//     Value b = param(b_data, "b");

//     // 3. Build a small network with checkpointed middle layer
//     //    y = ((x @ W) + b).relu()
//     Value y1 = matmul(x, W);
//     Value y2 = add(y1, b);

//     // Mark y2 as a checkpoint
//     y2 = checkpoint(y2);

//     // Apply activation
//     Value y3 = relu(y2);
//     Value loss = sum(y3);  // simple scalar loss

//     // 4. Backward pass
//     backward(loss);

//     // 5. Verify that checkpointed nodes recompute
//     std::cout << "\n--- Checkpoint verification ---\n";
//     auto n = y2.node;
//     if (n->is_checkpoint) {
//         std::cout << "Node " << n->debug_name << " is checkpointed ✅\n";
//     } else {
//         std::cout << "Node " << n->debug_name << " is NOT checkpointed ❌\n";
//     }

//     // 6. Inspect gradient values
//     std::cout << "\nGradients:\n";
//     std::cout << "dL/dW:\n" << W.grad() << "\n";
//     std::cout << "dL/db:\n" << b.grad() << "\n";

//     // 7. Check recomputation correctness manually
//     std::cout << "\nRecomputing checkpoint manually...\n";
//     bool recomputed = checkpoint_impl::recompute_subgraph(y2.node->shared_from_this());
//     std::cout << (recomputed ? "Recomputation success ✅\n" : "Recomputation failed ❌\n");

//     // 8. Print recomputed value
//     std::cout << "\nCheckpointed node value after recompute:\n";
//     std::cout << y2.node->value << "\n";

//     std::cout << "===== Test completed successfully =====\n";
//     return 0;
// }

// ============================================================
// File: test_auto_checkpoint.cpp
// Purpose: Verify automatic gradient checkpointing (every_n & by_depth)
// ============================================================

#include <iostream>
#include <vector>
#include "ad/ag_all.hpp"
#include "ad/checkpoint.hpp"
#include "ad/kernels_api.hpp"
#include <unordered_set>
#include <deque>

using namespace ag;

int main() {
    std::cout << "===== Auto Gradient Checkpointing Test =====\n";
    // ag::kernels::load_cpu_plugin("./libagkernels_cpu.so");
    // ------------------------------------------------------------
    // 1. Prepare small deterministic tensors
    Tensor x_data = Tensor::randn(2, 4, 42);
    Tensor W1_data = Tensor::randn(4, 4, 123);
    Tensor W2_data = Tensor::randn(4, 4, 321);
    Tensor W3_data = Tensor::randn(4, 2, 999);
    Tensor b1_data = Tensor::randn(1, 4, 55);
    Tensor b2_data = Tensor::randn(1, 4, 77);
    Tensor b3_data = Tensor::randn(1, 2, 88);

    // ------------------------------------------------------------
    // 2. Wrap them as Values
    Value x = constant(x_data, "x");
    Value W1 = param(W1_data, "W1");
    Value W2 = param(W2_data, "W2");
    Value W3 = param(W3_data, "W3");
    Value b1 = param(b1_data, "b1");
    Value b2 = param(b2_data, "b2");
    Value b3 = param(b3_data, "b3");

    // ------------------------------------------------------------
    // 3. Build a deeper network
    // y = relu((relu((x @ W1 + b1) @ W2 + b2)) @ W3 + b3)
    Value h1 = relu(add(matmul(x, W1), b1));
    Value h2 = relu(add(matmul(h1, W2), b2));
    Value y = add(matmul(h2, W3), b3);
    Value loss = sum(relu(y));  // scalar loss

    // ------------------------------------------------------------
    // 4. Apply automatic checkpointing
    std::cout << "\nApplying auto checkpointing...\n";
    auto_checkpoint_every_n(loss, 2);       // mark every 2nd node
    auto_checkpoint_by_depth(loss, 3);      // mark nodes deeper than depth 3

    // ------------------------------------------------------------
    // 5. Verify which nodes got checkpointed
    std::cout << "\n--- Auto checkpoint verification ---\n";
    int checkpointed_count = 0;
    std::deque<std::shared_ptr<Node>> q;
    std::unordered_set<Node*> visited;
    q.push_back(loss.node);

    while (!q.empty()) {
        auto n = q.front(); q.pop_front();
        if (!n || visited.count(n.get())) continue;
        visited.insert(n.get());
        if (n->is_checkpoint) {
            ++checkpointed_count;
            std::cout << "Checkpointed node: " << n->debug_name << " ✅\n";
        }
        for (auto &p : n->inputs)
            if (p) q.push_back(p);
    }

    if (checkpointed_count == 0)
        std::cout << "❌ No nodes were marked as checkpointed.\n";
    else
        std::cout << "✅ Total checkpointed nodes: " << checkpointed_count << "\n";

    // ------------------------------------------------------------
    // 6. Backward pass (triggers recomputation of checkpointed nodes)
    backward(loss);

    // ------------------------------------------------------------
    // 7. Inspect gradients for parameters
    std::cout << "\nGradients:\n";
    std::cout << "dL/dW1:\n" << W1.grad() << "\n";
    std::cout << "dL/dW2:\n" << W2.grad() << "\n";
    std::cout << "dL/dW3:\n" << W3.grad() << "\n";
    std::cout << "dL/db3:\n" << b3.grad() << "\n";

    // ------------------------------------------------------------
    // 8. Manual recomputation test on one of the checkpointed nodes
    std::cout << "\nManual recompute verification:\n";
    for (auto &n : visited) {
    if (n->is_checkpoint && !n->inputs.empty()) {
        bool ok = checkpoint_impl::recompute_subgraph(n->shared_from_this());
        std::cout << "Recomputed node (" << n->debug_name << "): "
                  << (ok ? "✅" : "❌") << "\n";
            break;
        }
    }


    std::cout << "\n===== Auto Checkpoint Test Completed =====\n";
    return 0;
}
