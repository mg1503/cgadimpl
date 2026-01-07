#include <iostream>
#include <chrono>
#include <iomanip>
#include <thread>
#include "ad/ag_all.hpp"

using namespace ag;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "PERFORMANCE BENCHMARK" << std::endl;
    std::cout << "Dependency Counter Parallelization" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int num_cores = std::thread::hardware_concurrency();
    std::cout << "\nSystem Info:" << std::endl;
    std::cout << "  CPU cores: " << num_cores << std::endl;
    std::cout << "\n⚡ TIP: Monitor CPU usage with 'htop' in another" << std::endl;
    std::cout << "   terminal to see all cores working during" << std::endl;
    std::cout << "   parallel backward pass!" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Create a large graph with 4 independent branches
    // Each branch has 1000 layers
    const int NUM_BRANCHES = 4;
    const int LAYERS_PER_BRANCH = 1000;
    const int HIDDEN_SIZE = 512;
    
    std::cout << "Building large computational graph..." << std::endl;
    std::cout << "  Branches: " << NUM_BRANCHES << std::endl;
    std::cout << "  Layers per branch: " << LAYERS_PER_BRANCH << std::endl;
    std::cout << "  Hidden size: " << HIDDEN_SIZE << std::endl;
    std::cout << "  Total operations: ~" << (NUM_BRANCHES * LAYERS_PER_BRANCH) << std::endl;
    
    // Create weights for each branch
    std::vector<Value> weights;
    for (int b = 0; b < NUM_BRANCHES; b++) {
        Tensor W = Tensor::randn<float>(Shape{{HIDDEN_SIZE, HIDDEN_SIZE}}, 
                                 TensorOptions().with_req_grad(true));
        Tensor B = Tensor::randn<float>(Shape{{1, HIDDEN_SIZE}}, 
                                 TensorOptions().with_req_grad(true));
        weights.push_back(make_tensor(W, ("w" + std::to_string(b)).c_str()));
    }
    
    // Create inputs for each branch
    std::vector<Value> branches;
    for (int b = 0; b < NUM_BRANCHES; b++) {
        Tensor X = Tensor::randn<float>(Shape{{1, HIDDEN_SIZE}}, 
                                TensorOptions().with_req_grad(false));
        auto x = make_tensor(X, ("x" + std::to_string(b)).c_str());
        
        // Build deep chain in this branch
        auto layer = x;
        for (int i = 0; i < LAYERS_PER_BRANCH; i++) {
            layer = relu(matmul(layer, weights[b]));
        }
        branches.push_back(layer);
        
        if (b % 1 == 0) {
            std::cout << "  ✓ Branch " << b << " built (" << LAYERS_PER_BRANCH << " layers)" << std::endl;
        }
    }
    
    // Merge all branches
    std::cout << "\nMerging branches..." << std::endl;
    auto result = branches[0];
    for (size_t i = 1; i < branches.size(); i++) {
        result = result + branches[i];
    }
    auto loss = sum(result);
    
    std::cout << "✓ Graph construction complete!" << std::endl;
    std::cout << "\n========================================" << std::endl;
    
    // ========================================
    // SEQUENTIAL BACKWARD
    // ========================================
    std::cout << "Running SEQUENTIAL backward pass..." << std::endl;
    std::cout << "(This will use only 1 CPU core)" << std::endl;
    
    zero_grad(loss);
    auto start_seq = std::chrono::high_resolution_clock::now();
    backward(loss, nullptr, false);
    auto end_seq = std::chrono::high_resolution_clock::now();
    auto duration_seq = std::chrono::duration_cast<std::chrono::milliseconds>(end_seq - start_seq);
    
    std::cout << "✅ Sequential backward completed!" << std::endl;
    std::cout << "   Time: " << duration_seq.count() << " ms" << std::endl;
    
    // ========================================
    // PARALLEL BACKWARD
    // ========================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running PARALLEL backward pass..." << std::endl;
    std::cout << "(This will use ALL " << num_cores << " CPU cores!)" << std::endl;
    std::cout << "👀 Watch your CPU monitor NOW!" << std::endl;
    
    zero_grad(loss);
    auto start_par = std::chrono::high_resolution_clock::now();
    backward(loss, nullptr, true);
    auto end_par = std::chrono::high_resolution_clock::now();
    auto duration_par = std::chrono::duration_cast<std::chrono::milliseconds>(end_par - start_par);
    
    std::cout << "✅ Parallel backward completed!" << std::endl;
    std::cout << "   Time: " << duration_par.count() << " ms" << std::endl;
    
    // ========================================
    // Results
    // ========================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "📊 PERFORMANCE RESULTS" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Sequential time:  " << std::setw(8) << duration_seq.count() << " ms" << std::endl;
    std::cout << "Parallel time:    " << std::setw(8) << duration_par.count() << " ms" << std::endl;
    
    if (duration_par.count() > 0) {
        double speedup = static_cast<double>(duration_seq.count()) / duration_par.count();
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "⚡ Speedup:        " << std::setw(8) << speedup << "x" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        if (speedup > 1.5) {
            std::cout << "\n🎉 Excellent speedup!" << std::endl;
            std::cout << "   The dependency counter successfully enabled" << std::endl;
            std::cout << "   parallel execution across multiple CPU cores!" << std::endl;
        } else {
            std::cout << "\nℹ️  Speedup is modest. This can happen because:" << std::endl;
            std::cout << "   1. Graph might not have enough parallelism" << std::endl;
            std::cout << "   2. Threading overhead can dominate for small ops" << std::endl;
            std::cout << "   3. Memory bandwidth can be a bottleneck" << std::endl;
        }
    }
    
    std::cout << "\n🎯 Key Insight:" << std::endl;
    std::cout << "   The dependency counter algorithm allows us to:" << std::endl;
    std::cout << "   • Process independent branches in parallel" << std::endl;
    std::cout << "   • Maintain correctness via atomic counters" << std::endl;
    std::cout << "   • Achieve speedups on multi-core CPUs" << std::endl;
    std::cout << "   • Scale to very large computation graphs" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
