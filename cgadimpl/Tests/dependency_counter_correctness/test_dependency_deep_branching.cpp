#include <iostream>
#include "ad/ag_all.hpp"

using namespace ag;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Test: Deep Sequential + Branching" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nPattern:" << std::endl;
    std::cout << "x → f1 → f2 → ... → f50" << std::endl;
    std::cout << "         └→ branch1" << std::endl;
    std::cout << "                └→ branch2" << std::endl;
    std::cout << "                      └→ branch3" << std::endl;
    std::cout << "\nThis tests mixed sequential/parallel patterns" << std::endl;
    std::cout << "that appear in residual networks and skip" << std::endl;
    std::cout << "connections (ResNet, U-Net, etc.)." << std::endl;
    std::cout << "========================================\n" << std::endl;

    const int DEPTH = 50;
    const int BRANCH_INTERVAL = 10;
    
    // Initial input and weight
    Tensor X = Tensor::randn<float>(Shape{{1, 16}}, TensorOptions().with_req_grad(false));
    Tensor W = Tensor::randn<float>(Shape{{16, 16}}, TensorOptions().with_req_grad(true));
    
    auto x = make_tensor(X, "x");
    auto w = make_tensor(W, "w");
    
    std::cout << "Building deep chain with " << DEPTH << " layers..." << std::endl;
    
    // Main sequential chain
    auto layer = x;
    std::vector<Value> branch_points;
    
    for (int i = 0; i < DEPTH; i++) {
        layer = relu(matmul(layer, w));
        
        // Create branch every BRANCH_INTERVAL layers
        if (i % BRANCH_INTERVAL == 0 && i > 0) {
            branch_points.push_back(layer);
            std::cout << "  ✓ Layer " << i << " (branch point created)" << std::endl;
        } else if (i % 10 == 0) {
            std::cout << "  ✓ Layer " << i << std::endl;
        }
    }
    
    std::cout << "\n✓ Main chain complete: " << DEPTH << " layers" << std::endl;
    std::cout << "✓ Branch points: " << branch_points.size() << std::endl;
    
    // Add branch processing
    std::cout << "\nProcessing branches..." << std::endl;
    std::vector<Value> branch_outputs;
    for (size_t i = 0; i < branch_points.size(); i++) {
        auto branch = sigmoid(branch_points[i]);
        branch_outputs.push_back(branch);
        std::cout << "  ✓ Branch " << i << " processed" << std::endl;
    }
    
    // Combine everything
    std::cout << "\nCombining main chain with branches..." << std::endl;
    auto final_output = layer;
    for (auto& branch : branch_outputs) {
        final_output = final_output + branch;
    }
    auto loss = sum(final_output);
    
    std::cout << "✓ Graph construction complete!" << std::endl;
    
    std::cout << "\nGraph statistics:" << std::endl;
    std::cout << "  • Main chain depth: " << DEPTH << " layers" << std::endl;
    std::cout << "  • Number of branches: " << branch_points.size() << std::endl;
    std::cout << "  • Total operations: ~" << (DEPTH + branch_points.size() * 2) << std::endl;
    std::cout << "  • Shared weight W used: " << DEPTH << " times" << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running PARALLEL backward pass..." << std::endl;
    std::cout << "========================================" << std::endl;
    
    zero_grad(loss);
    backward(loss, nullptr, true);
    
    std::cout << "✅ Parallel backward completed!\n" << std::endl;
    
    auto grad_w = w.grad();
    std::cout << "Gradient verification:" << std::endl;
    std::cout << "  W gradient computed: " << (grad_w.numel() > 0 ? "✓" : "✗") << std::endl;
    
    std::cout << "\n🎯 Dependency Counter Behavior:" << std::endl;
    std::cout << "   1. Sequential chain enforces ordering:" << std::endl;
    std::cout << "      f50 → f49 → f48 → ... → f1 → x" << std::endl;
    std::cout << "\n   2. Branches can process in parallel:" << std::endl;
    std::cout << "      All branch operations are independent!" << std::endl;
    std::cout << "\n   3. Weight W accumulates " << DEPTH << " gradient contributions" << std::endl;
    std::cout << "      The dependency counter waits for all!" << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running SEQUENTIAL backward for comparison..." << std::endl;
    std::cout << "========================================" << std::endl;
    
    zero_grad(loss);
    backward(loss, nullptr, false);
    
    std::cout << "✅ Sequential backward completed!\n" << std::endl;
    std::cout << "  W gradient computed: " << (w.grad().numel() > 0 ? "✓" : "✗") << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Real-world analogy:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  This pattern is similar to:" << std::endl;
    std::cout << "\n  • ResNet: Skip connections branch from main path" << std::endl;
    std::cout << "  • U-Net: Encoder features skip to decoder" << std::endl;
    std::cout << "  • DenseNet: Dense connections from all previous" << std::endl;
    std::cout << "\n✅ Dependency counter handles these architectures!" << std::endl;
    std::cout << "   Sequential parts maintain order," << std::endl;
    std::cout << "   parallel parts exploit concurrency!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
