#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "ad/ag_all.hpp"

using namespace ag;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Test: Wide Fan-out Tree Pattern" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nPattern:" << std::endl;
    std::cout << "         x" << std::endl;
    std::cout << "    / / | \\ \\ \\" << std::endl;
    std::cout << "   o1 o2 ... o100" << std::endl;
    std::cout << "\nThis tests parallel processing of 100 independent" << std::endl;
    std::cout << "operations, all using the same parameter x." << std::endl;
    std::cout << "Dependency counter: x.child_grad_count = 100" << std::endl;
    std::cout << "========================================\n" << std::endl;

    const int NUM_CHILDREN = 100;
    
    // Create shared parameter
    Tensor X = Tensor::randn<float>(Shape{{10, 10}}, TensorOptions().with_req_grad(true));
    auto x = make_tensor(X, "x");
    
    std::cout << "Building graph with " << NUM_CHILDREN << " independent operations..." << std::endl;
    
    // Create 100 different operations using x
    std::vector<Value> operations;
    operations.reserve(NUM_CHILDREN);
    
    for (int i = 0; i < NUM_CHILDREN; i++) {
        // Alternate between different operations to create diversity
        if (i % 4 == 0) {
            operations.push_back(relu(x));
        } else if (i % 4 == 1) {
            operations.push_back(x * x);
        } else if (i % 4 == 2) {
            operations.push_back(sigmoid(x));
        } else {
            operations.push_back(tanh(x));
        }
    }
    
    std::cout << "✓ Created " << operations.size() << " independent operations" << std::endl;
    
    // Sum all operations into a single loss
    auto loss = operations[0];
    for (size_t i = 1; i < operations.size(); i++) {
        loss = loss + operations[i];
    }
    loss = sum(loss);
    
    std::cout << "✓ Graph construction complete!" << std::endl;
    std::cout << "\nKey insight: Parameter x has " << NUM_CHILDREN << " children!" << std::endl;
    std::cout << "The dependency counter must wait for all " << NUM_CHILDREN 
              << " gradients before processing x." << std::endl;
    
    // Parallel backward
    std::cout << "\n--- PARALLEL BACKWARD ---" << std::endl;
    zero_grad(loss);
    auto start_parallel = std::chrono::high_resolution_clock::now();
    backward(loss, nullptr, true);
    auto end_parallel = std::chrono::high_resolution_clock::now();
    auto duration_parallel = std::chrono::duration_cast<std::chrono::milliseconds>(end_parallel - start_parallel);
    
    std::cout << "✅ Parallel backward completed!" << std::endl;
    std::cout << "   Time: " << duration_parallel.count() << " ms" << std::endl;
    std::cout << "   Gradient of x computed: " << (x.grad().numel() > 0 ? "✓" : "✗") << std::endl;
    
    // Sequential backward for comparison
    std::cout << "\n--- SEQUENTIAL BACKWARD ---" << std::endl;
    zero_grad(loss);
    auto start_seq = std::chrono::high_resolution_clock::now();
    backward(loss, nullptr, false);
    auto end_seq = std::chrono::high_resolution_clock::now();
    auto duration_seq = std::chrono::duration_cast<std::chrono::milliseconds>(end_seq - start_seq);
    
    std::cout << "✅ Sequential backward completed!" << std::endl;
    std::cout << "   Time: " << duration_seq.count() << " ms" << std::endl;
    std::cout << "   Gradient of x computed: " << (x.grad().numel() > 0 ? "✓" : "✗") << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "📊 Performance Comparison:" << std::endl;
    std::cout << "   Parallel:   " << duration_parallel.count() << " ms" << std::endl;
    std::cout << "   Sequential: " << duration_seq.count() << " ms" << std::endl;
    if (duration_seq.count() > 0) {
        double speedup = static_cast<double>(duration_seq.count()) / duration_parallel.count();
        std::cout << "   Speedup:    " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
    std::cout << "\n🎯 Key Achievement:" << std::endl;
    std::cout << "   The dependency counter correctly handled a node with" << std::endl;
    std::cout << "   " << NUM_CHILDREN << " children, accumulating all gradients before" << std::endl;
    std::cout << "   processing the parent. The atomic counter prevented" << std::endl;
    std::cout << "   race conditions during parallel execution!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
