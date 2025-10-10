#include "benchmark_utils.hpp"
#include <Eigen/Dense>

extern "C" {
    void matmul_impl_naive(const float*, const float*, float*, int, int, int);
    void matmul_impl_optimized(const float*, const float*, float*, int, int, int);
}

void benchmark_k(int K) {
    const int M = 256, N = 256;
    const int runs = 10;
    std::cout << "\n--- Benchmarking K dimension: " << M << "x" << K << "x" << N << " ---" << std::endl;

    std::vector<float> A(M * K), B(K * N), C(M * N);
    fill_random(A); fill_random(B);

    Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>> A_eigen(A.data(), M, K);
    Eigen::Map<Eigen::Matrix<float, -1, -1, Eigen::RowMajor>> B_eigen(B.data(), K, N);
    Eigen::Matrix<float, -1, -1, Eigen::RowMajor> C_eigen(M, N);
    auto eigen_func = [&](const float*, const float*, float*, int, int, int) {
        C_eigen.noalias() = A_eigen * B_eigen;
    };

    run_matmul_benchmark("Naive", matmul_impl_naive, A, B, C, M, K, N, runs);
    run_matmul_benchmark("Optimized", matmul_impl_optimized, A, B, C, M, K, N, runs);
    run_matmul_benchmark("Eigen", eigen_func, A, B, C, M, K, N, runs);
}

int main() {
    std::cout << "===== MatMul Cache Performance Benchmark (Varying K) =====" << std::endl;
    benchmark_k(16);
    benchmark_k(64);
    benchmark_k(256);
    benchmark_k(1024);
    benchmark_k(4096);
    return 0;
}