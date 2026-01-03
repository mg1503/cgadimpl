#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <unistd.h>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include "ad/ag_all.hpp"

using namespace ag;

// ==========================================
// Memory Measurement Helpers
// ==========================================
struct MemoryStats {
    long vm_peak_kb = 0;
    long vm_size_kb = 0;
    long vm_hwm_kb  = 0;
    long vm_rss_kb  = 0;
};

MemoryStats get_memory_stats() {
    MemoryStats stats;
    std::ifstream status_file("/proc/self/status");
    std::string line;
    if (!status_file.is_open()) return stats;
    while (std::getline(status_file, line)) {
        std::stringstream ss(line);
        std::string key;
        ss >> key;
        if (key == "VmPeak:") ss >> stats.vm_peak_kb;
        else if (key == "VmSize:") ss >> stats.vm_size_kb;
        else if (key == "VmHWM:")  ss >> stats.vm_hwm_kb;
        else if (key == "VmRSS:")  ss >> stats.vm_rss_kb;
    }
    return stats;
}

void print_mem_stats(const std::string& label, const MemoryStats& stats) {
    std::cout << std::left << std::setw(25) << label 
              << " | VmPeak: " << std::setw(8) << stats.vm_peak_kb << " KB"
              << " | VmRSS:  " << std::setw(8) << stats.vm_rss_kb << " KB" 
              << std::endl;
}

// ==========================================
// Benchmark Logic
// ==========================================
int main() {
    // 0. FIXED SEED
    srand(42); 
    
    std::cout << "========================================================\n";
    std::cout << "  BENCHMARK: Memory | Time | Precision (LINEAR)\n";
    std::cout << "========================================================\n\n";

    MemoryStats initial_stats = get_memory_stats();
    print_mem_stats("Baseline", initial_stats);

    // 2. Build Graph
    const int LAYERS = 50; 
    const int WIDTH = 1024;
    const int BATCH = 64;
    
    std::cout << "\n[Test Configuration]\n";
    std::cout << "  Layers: " << LAYERS << "\n";
    std::cout << "  Width:  " << WIDTH << "\n";
    std::cout << "  Init:   Deterministic (0.01)\n";
    std::cout << "  Model:  Linear (No Tanh)\n";

    auto start_build = std::chrono::high_resolution_clock::now();

    // Init inputs to small constant
    Tensor Xt = Tensor::full(Shape{{BATCH, WIDTH}}, TensorOptions().with_req_grad(true), 0.01f);
    Value x = make_tensor(Xt, "Input");
    
    Value h = x;
    std::vector<Value> params;
    
    for (int i = 0; i < LAYERS; ++i) {
        // Linear weights
        Tensor Wt = Tensor::full(Shape{{WIDTH, WIDTH}}, TensorOptions().with_req_grad(true), 0.01f);
        Value W = make_tensor(Wt, ("W" + std::to_string(i)).c_str());
        params.push_back(W);
        
        Value linear = matmul(h, W);
        
        // CRITICAL CHANGE: No Activation, just scaling to prevent explosion
        // Linear network ensures gradients flow all the way back
        h = linear * 0.1f; 
        
        if (i % 10 == 0 && i > 0) h = h + x;
    }
    
    Value loss = sum(h);
    
    auto end_build = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(end_build - start_build).count();
    
    MemoryStats build_stats = get_memory_stats();
    print_mem_stats("After Build", build_stats);

    // 3. Backward
    std::cout << "\n[Running Backward Pass...]\n";
    zero_grad(loss);
    auto start_back = std::chrono::high_resolution_clock::now();
    backward(loss); 
    auto end_back = std::chrono::high_resolution_clock::now();
    
    MemoryStats back_stats = get_memory_stats();
    print_mem_stats("After Backward", back_stats);
    
    double back_time = std::chrono::duration<double>(end_back - start_back).count();
    long mem_growth_kb = back_stats.vm_rss_kb - build_stats.vm_rss_kb;

    // 4. PRECISION CHECKSUM
    double input_grad_sum = 0.0;
    // Handle both Node-centric (node->grad) and Tensor-centric (node->tensor.grad_view())
    // We try to compile code that works for the user's current branch.
    // Since we are on 'main' (Node-centric), we use node->grad.
    // USER MUST MODIFY THIS LINE ON THE OTHER BRANCH if accessor differs.
    
    if (x.node) {
        // Try accessing via tensor first (works on both if node has tensor member)
        // Wait, main branch has `Tensor grad`, other might have `grad_view()`.
        // Let's use the most generic one for 'main' which is node->grad.
        // Actually, user updated it to `x.node->tensor.grad_view()` in Step 177.
        // I should stick to what compiles on THIS branch.
        // On 'main', Node has `Tensor grad`. So `x.node->grad`.
        
        // I will write the 'main' branch version.
        if (x.node->grad.numel() > 0) {
            const float* ptr = (const float*)x.node->grad.data();
            if (ptr) {
                 size_t n = std::min((size_t)1000, x.node->grad.numel());
                 for(size_t k=0; k<n; ++k) input_grad_sum += ptr[k];
            }
        }
    }

    double weight_grad_sum = 0.0;
    if (!params.empty()) {
        Value w0 = params[0];
        if (w0.node->grad.numel() > 0) {
            const float* ptr = (const float*)w0.node->grad.data();
             if (ptr) {
                 size_t n = std::min((size_t)1000, w0.node->grad.numel());
                 for(size_t k=0; k<n; ++k) weight_grad_sum += ptr[k];
            }
        }
    }
    
    double loss_val = 0.0;
    // Accessing value data
    // On main, Node has `Tensor value`.
    // On other branch, Node has `Tensor tensor`.
    // The user manually edited it to `node->tensor` previously.
    // I will write `node->value` as per 'main' definitions I saw earlier.
    // Wait, Step 133 shows `Tensor value;` in main branch graph.hpp.
    // So `node->value` is correct for main.
    
    if (loss.node->value.numel() > 0) {
        const float* ptr = (const float*)loss.node->value.data();
        if(ptr) loss_val = ptr[0];
    }

    std::cout << "\n========================================================\n";
    std::cout << "                  RESULTS SUMMARY\n";
    std::cout << "========================================================\n";
    std::cout << "  PERFORMANCE METRICS:\n";
    std::cout << "  --------------------\n";
    std::cout << "  Peak RSS (Physical)     | " << back_stats.vm_hwm_kb / 1024.0 << " MB\n";
    std::cout << "  Backward Execution Time | " << back_time << " s\n";
    std::cout << "  Memory Growth (Back)    | " << mem_growth_kb / 1024.0 << " MB\n";
    
    std::cout << "\n  PRECISION CHECKSUMS (Must Match Exactly):\n";
    std::cout << "  ----------------------------------------\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Final Loss Value        | " << loss_val << "\n";
    std::cout << "  Input X Grad Sum        | " << input_grad_sum << "\n";
    std::cout << "  Weight W0 Grad Sum      | " << weight_grad_sum << "\n";
    std::cout << "========================================================\n";

    return 0;
}
