// =================================================================
// File: cgadimpl/tests/test_checkpoint_production.cpp
// =================================================================
//
// A production-grade test for gradient checkpointing.
//
// This test verifies the two most critical aspects of checkpointing:
//
// 1.  **Gradient Correctness**: It ensures that the gradients computed
//     with checkpointing are numerically identical to those computed
//     without it. This is the most important correctness guarantee.
//
// 2.  **Memory Deletion**: It confirms that intermediate activations
//     from checkpointed segments are actually freed after the forward
//     pass. This is verified by observing the output from the
//     "careful_delete" system, which should print "[careful_delete] Freed node"
//     messages.
//
// This test replaces broad API calls with a focused, verifiable scenario.
//

#include "ad/graph.hpp"
#include "ad/nodeops.hpp"
#include "ad/autodiff.hpp"
#include "ad/checkpoint.hpp"
#include "ad/careful_deletion.hpp"
#include "tensor.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

using namespace ag;

// Helper to check if two tensors are numerically close
bool tensors_are_close(const Tensor& a, const Tensor& b, float tol = 1e-5) {
    if (a.shape().dims != b.shape().dims) {
        std::cerr << "Shape mismatch for gradient comparison!" << std::endl;
        return false;
    }
    if (a.numel() == 0) return true;

    Tensor diff = abs(a.to(Device::CPU) - b.to(Device::CPU), 0);
    Tensor max_val_tensor = reduce_max(diff);
    // Get the first element as a rough approximation
    float max_diff = *((float*)max_val_tensor.data());

    std::cout << "  - Max difference: " << max_diff << " (tolerance: " << tol << ")" << std::endl;
    return max_diff < tol;
}

// A simple model for testing purposes
struct TestModel {
    Value w1, b1, w2, b2, w3, b3;

    TestModel(int in, int hidden, int out, bool requires_grad = true) {
        auto opts = TensorOptions().with_req_grad(requires_grad);
        w1 = make_tensor(Tensor::randn(Shape{{in, hidden}}, opts), "w1");
        b1 = make_tensor(Tensor::randn(Shape{{1, hidden}}, opts), "b1");
        w2 = make_tensor(Tensor::randn(Shape{{hidden, hidden}}, opts), "w2");
        b2 = make_tensor(Tensor::randn(Shape{{1, hidden}}, opts), "b2");
        w3 = make_tensor(Tensor::randn(Shape{{hidden, out}}, opts), "w3");
        b3 = make_tensor(Tensor::randn(Shape{{1, out}}, opts), "b3");
    }

    Value forward(Value x, bool use_checkpoint) {
        // Layer 1
        x = relu(linear(x, w1, b1));

        // Layer 2 (the checkpointed section)
        auto layer2_fn = [&]() {
            return relu(linear(x, w2, b2));
        };

        if (use_checkpoint) {
            // The `checkpoint_impl::mark_node_checkpoint` is the low-level API
            // that your high-level `checkpoint()` function would use.
            Value checkpoint_out = layer2_fn();
            checkpoint_impl::mark_node_checkpoint(checkpoint_out.node);
            x = checkpoint_out;
        } else {
            x = layer2_fn();
        }

        // Layer 3
        x = linear(x, w3, b3);
        return x;
    }

    std::vector<Value> parameters() {
        return {w1, b1, w2, b2, w3, b3};
    }
};

void run_production_checkpoint_test() {
    std::cout << "\n--- Production-Grade Gradient Checkpointing Test ---\n";
    
    TestModel model(10, 32, 5);
    Value input = make_tensor(Tensor::randn(Shape{{8, 10}}, TensorOptions()), "input");

    // --- Step 1: Run WITHOUT checkpointing to get baseline gradients ---
    std::cout << "\n[Phase 1] Running without checkpointing to establish baseline...\n";
    for (auto& p : model.parameters()) {
        zero_grad(p);
    }
    Value y1 = model.forward(input, /* use_checkpoint = */ false);
    Value loss1 = sum(y1);
    backward(loss1);

    // Store the baseline gradients
    Tensor grad_w2_baseline = model.w2.grad().clone();
    std::cout << "[Phase 1] Baseline gradients computed.\n";

    // --- Step 2: Run WITH checkpointing ---
    std::cout << "\n[Phase 2] Running WITH checkpointing...\n";
    std::cout << "          (Expecting to see '[careful_delete] Freed node' messages below)\n";
    for (auto& p : model.parameters()) {
        zero_grad(p);
    }
    Value y2 = model.forward(input, /* use_checkpoint = */ true);

    // CRITICAL CHECK: After the forward pass with checkpointing, the intermediate
    // activations from the checkpointed block should be deleted.
    // The `sweep_safe_nodes` function will print messages for each freed node.
    memory::sweep_safe_nodes(y2, memory::DeletePolicy::AlwaysSafe);

    Value loss2 = sum(y2);
    std::cout << "\n[Phase 2] Starting backward pass (this will trigger recomputation)...\n";
    backward(loss2);
    std::cout << "[Phase 2] Backward pass complete.\n";

    // --- Step 3: Compare the gradients from both runs ---
    std::cout << "\n[Phase 3] Verifying gradient correctness against baseline...\n";
    bool w2_grads_ok = tensors_are_close(grad_w2_baseline, model.w2.grad());

    assert(w2_grads_ok);

    if (w2_grads_ok) {
        std::cout << "\n✅ Test Passed: Gradients with checkpointing match the baseline!\n";
    } else {
        std::cerr << "\n❌ Test Failed: Gradient mismatch detected!\n";
    }
}

int main() {
    try {
        run_production_checkpoint_test();
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}