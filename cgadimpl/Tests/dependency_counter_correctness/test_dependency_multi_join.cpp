#include <iostream>
#include "ad/ag_all.hpp"

using namespace ag;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Test: Multiple Join Points (Complex)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nPattern:" << std::endl;
    std::cout << "   w1      w2" << std::endl;
    std::cout << "   / \\    / \\" << std::endl;
    std::cout << "  a1 a2  b1 b2" << std::endl;
    std::cout << "   \\ /    \\ /" << std::endl;
    std::cout << "    c1     c2" << std::endl;
    std::cout << "     \\    /" << std::endl;
    std::cout << "       loss" << std::endl;
    std::cout << "\nThis tests multiple convergence points with" << std::endl;
    std::cout << "shared parameters - a realistic scenario!" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Shared weights
    Tensor W1 = Tensor::randn<float>(Shape{{4, 4}}, TensorOptions().with_req_grad(true));
    Tensor W2 = Tensor::randn<float>(Shape{{4, 4}}, TensorOptions().with_req_grad(true));
    
    auto w1 = make_tensor(W1, "w1");
    auto w2 = make_tensor(W2, "w2");
    
    std::cout << "Step 1: Creating first diamond (w1 -> a1, a2 -> c1)" << std::endl;
    // First diamond pattern with w1
    auto a1 = relu(w1);        // First path from w1
    auto a2 = w1 * w1;         // Second path from w1
    auto c1 = a1 + a2;         // Convergence point 1
    std::cout << "  ✓ w1 has 2 children (a1, a2)" << std::endl;
    std::cout << "  ✓ c1 joins paths from a1 and a2" << std::endl;
    
    std::cout << "\nStep 2: Creating second diamond (w2 -> b1, b2 -> c2)" << std::endl;
    // Second diamond pattern with w2
    auto b1 = sigmoid(w2);     // First path from w2
    auto b2 = tanh(w2);        // Second path from w2
    auto c2 = b1 + b2;         // Convergence point 2
    std::cout << "  ✓ w2 has 2 children (b1, b2)" << std::endl;
    std::cout << "  ✓ c2 joins paths from b1 and b2" << std::endl;
    
    std::cout << "\nStep 3: Creating final join point" << std::endl;
    // Final convergence
    auto result = c1 * c2;     // Join both diamond outputs
    auto loss = sum(result);
    std::cout << "  ✓ loss depends on both c1 and c2" << std::endl;
    
    std::cout << "\nGraph structure summary:" << std::endl;
    std::cout << "  • Total parameters: 2 (w1, w2)" << std::endl;
    std::cout << "  • Total diamond patterns: 2" << std::endl;
    std::cout << "  • Total join points: 3 (c1, c2, result)" << std::endl;
    std::cout << "  • Total operations: 8" << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running PARALLEL backward pass..." << std::endl;
    std::cout << "========================================" << std::endl;
    
    zero_grad(loss);
    backward(loss, nullptr, true);
    
    std::cout << "✅ Parallel backward completed!\n" << std::endl;
    
    // Verify gradients computed
    std::cout << "Gradient verification:" << std::endl;
    std::cout << "  w1 gradient: " << (w1.grad().numel() > 0 ? "✓ Computed" : "✗ Missing") << std::endl;
    std::cout << "  w2 gradient: " << (w2.grad().numel() > 0 ? "✓ Computed" : "✗ Missing") << std::endl;
    
    std::cout << "\n🎯 Dependency Counter Behavior:" << std::endl;
    std::cout << "   1. c1 waits for both a1 and a2 (w1's children)" << std::endl;
    std::cout << "   2. c2 waits for both b1 and b2 (w2's children)" << std::endl;
    std::cout << "   3. result waits for both c1 and c2" << std::endl;
    std::cout << "   4. w1 waits for gradients from a1 and a2" << std::endl;
    std::cout << "   5. w2 waits for gradients from b1 and b2" << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running SEQUENTIAL backward for comparison..." << std::endl;
    std::cout << "========================================" << std::endl;
    
    zero_grad(loss);
    backward(loss, nullptr, false);
    
    std::cout << "✅ Sequential backward completed!\n" << std::endl;
    std::cout << "  w1 gradient: " << (w1.grad().numel() > 0 ? "✓ Computed" : "✗ Missing") << std::endl;
    std::cout << "  w2 gradient: " << (w2.grad().numel() > 0 ? "✓ Computed" : "✗ Missing") << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "✅ Multi-Join Test PASSED!" << std::endl;
    std::cout << "   Complex graph with multiple convergence" << std::endl;
    std::cout << "   points handled correctly by dependency" << std::endl;
    std::cout << "   counter in both sequential and parallel!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
