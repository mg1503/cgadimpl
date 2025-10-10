#include "benchmark_utils.hpp"
#include <Eigen/Dense>

extern "C" {
    void matmul_impl_naive(const float*, const float*, float*, int, int, int);
    void matmul_impl_optimized(const float*, const float*, float*, int, int, int);
}

void benchmark_size(const std::string& title, int M, int K, int N, int runs) {
    std::cout << "\n--- " << title << ": " << M << "x" << K << "x" << N << " (" << runs << " runs) ---" << std::endl;
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
    std::cout << "===== MatMul Aspect Ratio Benchmark =====" << std::endl;
    // Tall & Skinny: M is large, K and N are small
    benchmark_size("Tall & Skinny", 4096, 64, 64, 10);
    // Fat & Short: K is large, M and N are small
    benchmark_size("Fat & Short", 64, 4096, 64, 10);
    // Outer Product: K=1
    benchmark_size("Outer Product", 2048, 1, 2048, 10);
    return 0;
}