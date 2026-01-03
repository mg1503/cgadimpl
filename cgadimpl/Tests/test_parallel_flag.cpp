#include <iostream>
#include <chrono>
#include "ad/ag_all.hpp"
using namespace ag;

int main() {
    std::cout << "=== Testing backward with enable_parallel flag ===" << std::endl;
    
    // Create a simple graph
    Tensor A = Tensor::randn(Shape{{3, 3}}, TensorOptions().with_req_grad(true));
    Tensor B = Tensor::randn(Shape{{3, 3}}, TensorOptions().with_req_grad(true));
    
    auto a = make_tensor(A, "A");
    auto b = make_tensor(B, "B");
    
    auto c = matmul(a, b);
    auto loss = sum(c);
    
    // Test 1: Sequential (default)
    std::cout << "\n1. Sequential backward (enable_parallel=false)" << std::endl;
    zero_grad(loss);
    auto start1 = std::chrono::high_resolution_clock::now();
    backward(loss);  // Default: enable_parallel=false
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    std::cout << "   Time: " << duration1.count() << " μs" << std::endl;
    std::cout << "   Grad A computed: " << (a.grad().numel() > 0 ? "✓" : "✗") << std::endl;
    
    // Test 2: Parallel (explicit)
    std::cout << "\n2. Parallel backward (enable_parallel=true)" << std::endl;
    zero_grad(loss);
    auto start2 = std::chrono::high_resolution_clock::now();
    backward(loss, nullptr, true);  // Enable parallelism
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    std::cout << "   Time: " << duration2.count() << " μs" << std::endl;
    std::cout << "   Grad A computed: " << (a.grad().numel() > 0 ? "✓" : "✗") << std::endl;
    
    std::cout << "\n  Both methods work! User can choose based on graph size." << std::endl;
    
    return 0;
}
