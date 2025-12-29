#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <unistd.h>
#include <iomanip>
#include <numeric>
#include "ad/ag_all.hpp"

using namespace ag;

// ==========================================
// Memory Measurement Helpers (Linux Specific)
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
    // 0. FIXED SEED for Precision Comparison
    srand(42); 
    // If your Tensor library has a global seed, set it here explicitly if possible.
    // For now, assuming randn uses std::rand or similar common source from <random> 
    // initialized often in main. If cgadimpl has explicit seeder, use it.
    
    std::cout << "========================================================\n";
    std::cout << "  BENCHMARK: Memory | Time | Precision (Fixed Seed)\n";
    std::cout << "========================================================\n\n";

    MemoryStats initial_stats = get_memory_stats();
    print_mem_stats("Baseline", initial_stats);

    // 2. Build Graph
    const int LAYERS = 200;
    const int WIDTH = 1024;
    const int BATCH = 64;
    
    auto start_build = std::chrono::high_resolution_clock::now();

    // Use a simple deterministic initialization if random is inconsistent across branches
    // But ideally randn is consistent if seeded. 
    // Let's use `full` or `ones` modulated to be safe if we rely on checksums?
    // Actually, `randn` is better for creating non-trivial gradients. 
    // We assume the user's random implementation is consistent.
    
    Tensor Xt = Tensor::randn(Shape{{BATCH, WIDTH}}, TensorOptions().with_req_grad(true));
    Value x = make_tensor(Xt, "Input");
    
    Value h = x;
    std::vector<Value> params;
    
    for (int i = 0; i < LAYERS; ++i) {
        Tensor Wt = Tensor::randn(Shape{{WIDTH, WIDTH}}, TensorOptions().with_req_grad(true));
        Value W = make_tensor(Wt, ("W" + std::to_string(i)).c_str());
        params.push_back(W);
        
        Value linear = matmul(h, W);
        h = tanh(linear); 
        if (i % 10 == 0 && i > 0) h = h + x;
    }
    
    Value loss = sum(h);
    
    auto end_build = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(end_build - start_build).count();
    
    MemoryStats build_stats = get_memory_stats();
    print_mem_stats("After Build", build_stats);

    // 3. Backward
    zero_grad(loss);
    auto start_back = std::chrono::high_resolution_clock::now();
    backward(loss); 
    auto end_back = std::chrono::high_resolution_clock::now();
    
    MemoryStats back_stats = get_memory_stats();
    print_mem_stats("After Backward", back_stats);
    
    double back_time = std::chrono::duration<double>(end_back - start_back).count();
    long mem_growth_kb = back_stats.vm_rss_kb - build_stats.vm_rss_kb;

    // 4. PRECISION CHECKSUM
    // Sum up the gradients of the Input X and the first Weight W0
    // This allows verifying that gradients are identical to the 6th decimal independent of approach.
    
    double input_grad_sum = 0.0;
    if (x.node && x.node->tensor.grad_view().numel() > 0) {
        // Access data safely
        // Assuming CPU float pointer
        const float* ptr = (const float*)x.node->tensor.grad_view().data();
        if (ptr) {
             for(size_t k=0; k<x.node->tensor.grad_view().numel(); ++k) input_grad_sum += ptr[k];
        }
    }

    double weight_grad_sum = 0.0;
    if (!params.empty()) {
        Value w0 = params[0];
        const float* ptr = (const float*)w0.node->tensor.grad_view().data();
        if (ptr) {
             for(size_t k=0; k<w0.node->tensor.grad_view().numel(); ++k) weight_grad_sum += ptr[k];
        }
    }
    
    double loss_val = 0.0;
    if (loss.node->tensor.numel() > 0) {
        const float* ptr = (const float*)loss.node->tensor.data();
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
