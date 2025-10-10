#include "benchmark_utils.hpp"
#include <Eigen/Dense>

extern "C" {
    void matmul_impl_naive(const float*, const float*, float*, int, int, int);
    void matmul_impl_optimized(const float*, const float*, float*, int, int, int);
}

void benchmark_latency(int M, int K, int N, int runs) {
    std::cout << "\n--- Benchmarking Latency: " << M << "x" << K << "x" << N << " (" << runs << " runs) ---" << std::endl;
    std::vector<float> A(M * K), B(K * N), C(M * N);
    fill_random(A);
    fill_random(B);

    Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>> A_eigen(A.data(), M, K);
    Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>> B_eigen(B.data(), K, N);
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> C_eigen(M, N);
    auto eigen_func = [&](const float*, const float*, float*, int, int, int) {
        C_eigen.noalias() = A_eigen * B_eigen;
    };

    // Note: We modify run_matmul_benchmark's output for microseconds
    auto run_latency = [&](const std::string& name, auto func) {
        Timer timer;
        double total_us = 0;
        func(A.data(), B.data(), C.data(), M, K, N); // Warm-up
        for (int i = 0; i < runs; ++i) {
            timer.start();
            func(A.data(), B.data(), C.data(), M, K, N);
            total_us += timer.stop() * 1000.0; // convert ms to us
        }
        std::cout << std::left << std::setw(12) << name << ": "
                  << std::fixed << std::setprecision(3) << std::setw(10)
                  << (total_us / runs) << " us" << std::endl;
    };

    run_latency("Naive", matmul_impl_naive);
    run_latency("Optimized", matmul_impl_optimized);
    run_latency("Eigen", eigen_func);
}

int main() {
    std::cout << "===== MatMul Latency Benchmark =====" << std::endl;
    benchmark_latency(8, 8, 8, 100000);
    benchmark_latency(16, 16, 16, 50000);
    benchmark_latency(32, 32, 32, 10000);
    return 0;
}