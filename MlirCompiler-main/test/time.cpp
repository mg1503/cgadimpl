#include <chrono>
#include <iostream>
#include <cstdint>
#include <cmath>
#include <iomanip>

extern "C" {

    /**
     * @brief Returns the current time in microseconds.
     */
    int64_t get_time() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    }
    
    /**
     * @brief Calculates and prints GFLOPS given the matrix dimensions and execution time.
     * The dimensions are passed from the MLIR code for dynamic calculation.
     * * @param m Matrix dimension M (rows of A, rows of C)
     * @param n Matrix dimension N (cols of B, cols of C)
     * @param k Matrix dimension K (cols of A, rows of B)
     * @param time_us The execution time in microseconds.
     */
    void print_gflops(int64_t m, int64_t n, int64_t k, int64_t time_us) {
        if (time_us <= 0) {
            std::cerr << "Error: Execution time is zero or negative." << std::endl;
            return;
        }

        // FLOPS calculation: 2 * M * N * K for matrix multiplication
        // The MLIR also includes the element-wise bias addition, which is M*N additional ops.
        // We focus on the dominant MatMul FLOPS here.
        double flops = 2.0 * (double)m * (double)n * (double)k;  
        double time_sec = (double)time_us / 1e6;
        
        // GFLOPS = FLOPS / (Time in seconds * 10^9)
        double gflops = flops / (time_sec * 1e9);

        std::cout << "\n--- Performance Summary ---\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Matrix Size: " << m << "x" << k << " * " << k << "x" << n << std::endl;
        std::cout << "Execution Time: " << time_us << " microseconds (" << time_sec << "s)" << std::endl;
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
        std::cout << "---------------------------\n" << std::endl;
    }
}