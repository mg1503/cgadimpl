#include "ad/core/graph.hpp"
#include "ad/ops/nodeops.hpp"
#include <iostream>
#include <vector>

/**
 * @file test_nodiscard.cpp
 * @brief Demonstrates how AG_NODISCARD catches logic errors in an MLP model.
 * 
 * This file contains an MLP implementation with intentional "mistakes" where
 * the results of operations are ignored. This should trigger compiler warnings.
 */

using namespace ag;
using namespace ag::detail;

void run_mistaken_mlp() {
    std::cout << "Initializing Mistaken MLP..." << std::endl;

    // 1. Setup inputs and weights
    Tensor x_val = Tensor::randn(Shape{{1, 10}}, TensorOptions());
    Tensor w1_val = Tensor::randn(Shape{{10, 5}}, TensorOptions());
    Tensor b1_val = Tensor::randn(Shape{{1, 5}}, TensorOptions());
    Tensor target_val = Tensor::randn(Shape{{1, 5}}, TensorOptions());

    Value x = make_tensor(x_val, "input");
    Value w1 = make_tensor(w1_val, "w1");
    Value b1 = make_tensor(b1_val, "b1");
    Value target = make_tensor(target_val, "target");

    // ========================================================================
    // INTENTIONAL MISTAKES (Ignoring return values)
    // ========================================================================

    // Mistake 1: Ignoring the result of matmul
    // Rationale: The user might think matmul_nodeops modifies 'x' in-place,
    // but it actually returns a new node.
    matmul_nodeops(x.node, w1.node); 

    // Mistake 2: Ignoring the result of addition
    // Rationale: Forgetting to capture the bias addition.
    add_nodeops(x.node, b1.node);

    // Mistake 3: Ignoring the activation function
    // Rationale: Forgetting to apply ReLU or thinking it's in-place.
    relu_nodeops(x.node);

    // Mistake 4: Ignoring the loss calculation
    // Rationale: The training loop won't have a loss to backpropagate from.
    mse_loss_nodeops(x.node, target.node);

    std::cout << "Mistaken MLP 'execution' finished (but logic was ignored)." << std::endl;
}

int main() {
    run_mistaken_mlp();
    return 0;
}
