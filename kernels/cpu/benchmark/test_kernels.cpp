// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <numeric>
// #include <random>

// // This include will work once we configure the CMakeLists.txt correctly
// #include "ad/kernels_api.hpp"

// // Use Eigen for a highly optimized reference implementation
// #include <Eigen/Dense>

// // Forward declare the functions from your agkernels_cpu.cpp file
// // so this test file knows they exist. The linker will connect them later.
// extern "C" {
//     void relu_impl(const float* x, float* y, int64_t n);
//     void matmul_impl(const float* A, const float* B, float* C, int M, int K, int N);
// }

// // Helper function to generate random data
// void fill_random(std::vector<float>& vec) {
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
//     for (auto& v : vec) {
//         v = dis(gen);
//     }
// }

// int main() {
//     // --- ReLU Benchmark ---
//     std::cout << "--- Benchmarking ReLU ---" << std::endl;
//     const int64_t n_relu = 10000000; // 10 million elements
//     const int num_relu_runs = 100;
//     std::vector<float> x_relu(n_relu);
//     std::vector<float> y_relu(n_relu);
//     fill_random(x_relu);

//     // Your ReLU implementation
//     auto start_your_relu = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < num_relu_runs; ++i) {
//         relu_impl(x_relu.data(), y_relu.data(), n_relu);
//     }
//     auto end_your_relu = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> your_relu_duration = (end_your_relu - start_your_relu) / num_relu_runs;
//     std::cout << "Your relu_impl average time: " << your_relu_duration.count() << " ms" << std::endl;

//     // Eigen ReLU implementation
//     Eigen::Map<Eigen::VectorXf> x_eigen_relu(x_relu.data(), n_relu);
//     Eigen::VectorXf y_eigen_relu(n_relu);
//     auto start_eigen_relu = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < num_relu_runs; ++i) {
//         y_eigen_relu = x_eigen_relu.cwiseMax(0.f);
//     }
//     auto end_eigen_relu = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> eigen_relu_duration = (end_eigen_relu - start_eigen_relu) / num_relu_runs;
//     std::cout << "Eigen ReLU average time: " << eigen_relu_duration.count() << " ms" << std::endl;
//     std::cout << std::endl;


//     // --- MatMul Benchmark ---
//     std::cout << "--- Benchmarking MatMul ---" << std::endl;
//     const int M = 512;
//     const int K = 512;
//     const int N = 512;
//     const int num_matmul_runs = 10;
//     std::vector<float> A(M * K);
//     std::vector<float> B(K * N);
//     std::vector<float> C(M * N);
//     fill_random(A);
//     fill_random(B);

//     // Your MatMul implementation
//     auto start_your_matmul = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < num_matmul_runs; ++i) {
//         matmul_impl(A.data(), B.data(), C.data(), M, K, N);
//     }
//     auto end_your_matmul = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> your_matmul_duration = (end_your_matmul - start_your_matmul) / num_matmul_runs;
//     std::cout << "Your matmul_impl average time: " << your_matmul_duration.count() << " ms" << std::endl;

//     // Eigen MatMul implementation (Row-major for direct comparison)
//     Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_eigen(A.data(), M, K);
//     Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B_eigen(B.data(), K, N);
//     Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> C_eigen(M, N);

//     auto start_eigen_matmul = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < num_matmul_runs; ++i) {
//         C_eigen.noalias() = A_eigen * B_eigen;
//     }
//     auto end_eigen_matmul = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> eigen_matmul_duration = (end_eigen_matmul - start_eigen_matmul) / num_matmul_runs;
//     std::cout << "Eigen MatMul average time: " << eigen_matmul_duration.count() << " ms" << std::endl;

//     return 0;
// }
// ================================================================================================================================================================================================
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <random>

// This include will work because of the CMake configuration
#include "ad/kernels_api.hpp"

// Use Eigen for a highly optimized reference implementation
#include <Eigen/Dense>

// Forward declare the functions from your agkernels_cpu.cpp file
// *** IMPORTANT: These names must EXACTLY match the function names in the .cpp file ***
extern "C" {
    void relu_impl_optimized(const float* x, float* y, int64_t n); // <-- CHANGED
    void matmul_impl_optimized(const float* A, const float* B, float* C, int M, int K, int N); // <-- CHANGED
}

// Helper function to generate random data
void fill_random(std::vector<float>& vec) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& v : vec) {
        v = dis(gen);
    }
}

int main() {
    // --- ReLU Benchmark ---
    std::cout << "--- Benchmarking ReLU ---" << std::endl;
    const int64_t n_relu = 10000000; // 10 million elements
    const int num_relu_runs = 100;
    std::vector<float> x_relu(n_relu);
    std::vector<float> y_relu(n_relu);
    fill_random(x_relu);

    // Your optimized ReLU implementation
    auto start_your_relu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_relu_runs; ++i) {
        // Call the new function name
        relu_impl_optimized(x_relu.data(), y_relu.data(), n_relu); // <-- CHANGED
    }
    auto end_your_relu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> your_relu_duration = (end_your_relu - start_your_relu) / num_relu_runs;
    std::cout << "Our optimized relu_impl average time: " << your_relu_duration.count() << " ms" << std::endl;

    // Eigen ReLU implementation
    Eigen::Map<Eigen::VectorXf> x_eigen_relu(x_relu.data(), n_relu);
    Eigen::VectorXf y_eigen_relu(n_relu);
    auto start_eigen_relu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_relu_runs; ++i) {
        y_eigen_relu = x_eigen_relu.cwiseMax(0.f);
    }
    auto end_eigen_relu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> eigen_relu_duration = (end_eigen_relu - start_eigen_relu) / num_relu_runs;
    std::cout << "Eigen ReLU average time: " << eigen_relu_duration.count() << " ms" << std::endl;
    std::cout << std::endl;


    // --- MatMul Benchmark ---
    std::cout << "--- Benchmarking MatMul ---" << std::endl;
    const int M = 512;
    const int K = 512;
    const int N = 512;
    const int num_matmul_runs = 10;
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);
    fill_random(A);
    fill_random(B);

    // Your optimized MatMul implementation
    auto start_your_matmul = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_matmul_runs; ++i) {
        // Call the new function name
        matmul_impl_optimized(A.data(), B.data(), C.data(), M, K, N); // <-- CHANGED
    }
    auto end_your_matmul = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> your_matmul_duration = (end_your_matmul - start_your_matmul) / num_matmul_runs;
    std::cout << "Our optimized matmul_impl average time: " << your_matmul_duration.count() << " ms" << std::endl;

    // Eigen MatMul implementation (Row-major for direct comparison)
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_eigen(A.data(), M, K);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B_eigen(B.data(), K, N);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> C_eigen(M, N);

    auto start_eigen_matmul = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_matmul_runs; ++i) {
        C_eigen.noalias() = A_eigen * B_eigen;
    }
    auto end_eigen_matmul = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> eigen_matmul_duration = (end_eigen_matmul - start_eigen_matmul) / num_matmul_runs;
    std::cout << "Eigen MatMul average time: " << eigen_matmul_duration.count() << " ms" << std::endl;

    return 0;
}
