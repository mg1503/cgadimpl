#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <unistd.h>
#include <iomanip>
#include "ad/ag_all.hpp"

using namespace ag;

// ==========================================
// Memory Measurement Helpers (Linux Specific)
// ==========================================
struct MemoryStats {
    long vm_peak_kb = 0; // Peak virtual memory size
    long vm_size_kb = 0; // Current virtual memory size
    long vm_hwm_kb  = 0; // Peak resident set size ("High Water Mark")
    long vm_rss_kb  = 0; // Current resident set size
};

MemoryStats get_memory_stats() {
    MemoryStats stats;
    std::ifstream status_file("/proc/self/status");
    std::string line;
    
    if (!status_file.is_open()) {
        std::cerr << "Warning: Could not open /proc/self/status" << std::endl;
        return stats;
    }

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
int main(int argc, char** argv) {
    std::cout << "========================================================\n";
    std::cout << "     GRADIENT APPROACH MEMORY BENCHMARK\n";
    std::cout << "========================================================\n\n";

    // 1. Initial Baseline
    MemoryStats initial_stats = get_memory_stats();
    print_mem_stats("Baseline (Start)", initial_stats);

    // 2. Build a Heavy Graph
    // Deep MLP with wide layers to stress memory allocation
    const int LAYERS = 200;
    const int WIDTH = 1024;
    const int BATCH = 64;
    
    std::cout << "\n[Test Configuration]\n";
    std::cout << "  Layers: " << LAYERS << "\n";
    std::cout << "  Width:  " << WIDTH << "\n";
    std::cout << "  Batch:  " << BATCH << "\n";
    
    auto start_build = std::chrono::high_resolution_clock::now();

    Tensor Xt = Tensor::randn(Shape{{BATCH, WIDTH}}, TensorOptions().with_req_grad(false));
    Value x = make_tensor(Xt, "Input");
    
    // Create shared weights to reduce parameter count but keep activation memory high
    // (Simulating a Recurrent-like structure or just deep shared weights)
    // Actually, distinct weights stress memory MORE, so let's make distinct weights
    // But to avoid OOM on small machines, maybe just a few sets reused? 
    // Let's use distinct weights for max stress.
    
    std::vector<Value> params;
    Value h = x;
    
    for (int i = 0; i < LAYERS; ++i) {
        // Linear layer: W[WIDTH, WIDTH]
        Tensor Wt = Tensor::randn(Shape{{WIDTH, WIDTH}}, TensorOptions().with_req_grad(true));
        Value W = make_tensor(Wt, ("W" + std::to_string(i)).c_str());
        params.push_back(W);
        
        // Operation: h = tanh(h @ W)
        Value linear = matmul(h, W);
        h = tanh(linear); 
        
        // Occasional skip connection to make graph interesting
        if (i % 10 == 0 && i > 0) {
             h = h + x; // Broadcast add or matching shape add
        }
    }
    
    Value loss = sum(h);
    
    auto end_build = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(end_build - start_build).count();
    
    MemoryStats build_stats = get_memory_stats();
    print_mem_stats("After Graph Build", build_stats);
    std::cout << "  Build Time: " << build_time << " s\n";

    // 3. Backward Pass (The Critical Part)
    std::cout << "\n[Running Backward Pass...]\n";
    
    zero_grad(loss);
    
    auto start_back = std::chrono::high_resolution_clock::now();
    backward(loss); // Standard sequential backward
    auto end_back = std::chrono::high_resolution_clock::now();
    
    MemoryStats back_stats = get_memory_stats();
    print_mem_stats("After Backward", back_stats);
    
    double back_time = std::chrono::duration<double>(end_back - start_back).count();
    std::cout << "  Backward Time: " << back_time << " s\n";

    // 4. Metrics Calculation
    long mem_growth_kb = back_stats.vm_rss_kb - build_stats.vm_rss_kb;
    double mem_growth_mb = mem_growth_kb / 1024.0;
    
    // Estimate Allocation Rate based on Resident Memory Growth / Time
    // (This is a lower bound, as it ignores temporary allocations that were freed)
    double alloc_rate_mbs = mem_growth_mb / back_time;

    std::cout << "\n========================================================\n";
    std::cout << "                  RESULTS SUMMARY\n";
    std::cout << "========================================================\n";
    std::cout << "  Metric                      | Value\n";
    std::cout << "  --------------------------- | --------------------\n";
    std::cout << "  Peak Virtual Memory         | " << back_stats.vm_peak_kb / 1024.0 << " MB\n";
    std::cout << "  Peak RSS (Physical)         | " << back_stats.vm_hwm_kb / 1024.0 << " MB\n";
    std::cout << "  Backward Memory Growth      | " << mem_growth_mb << " MB\n";
    std::cout << "  Backward Execution Time     | " << back_time << " s\n";
    std::cout << "  Approx. Net Alloc Rate      | " << alloc_rate_mbs << " MB/s\n";
    std::cout << "========================================================\n";
    
    std::cout << "\nINTERPRETATION GUIDE:\n";
    std::cout << "1. Run this binary on Branch A (Node-centric).\n";
    std::cout << "2. Checkout Branch B (Tensor-centric).\n";
    std::cout << "3. Recompile and run this binary again.\n";
    std::cout << "4. Compare 'Backward Memory Growth' and 'Peak RSS'.\n";
    std::cout << "   - Lower Peak RSS = Better Memory Utilization.\n";
    std::cout << "   - Lower Execution Time = Better Throughput.\n";
    
    return 0;
}
