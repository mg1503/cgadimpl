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
// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <numeric>
// #include <random>

// // This include will work because of the CMake configuration
// #include "ad/kernels_api.hpp"

// // Use Eigen for a highly optimized reference implementation
// #include <Eigen/Dense>

// // Forward declare the functions from your agkernels_cpu.cpp file
// // *** IMPORTANT: These names must EXACTLY match the function names in the .cpp file ***
// extern "C" {
//     void relu_impl_optimized(const float* x, float* y, int64_t n); // <-- CHANGED
//     void matmul_impl_optimized(const float* A, const float* B, float* C, int M, int K, int N); // <-- CHANGED
//     void gelu_impl_optimized(const float* x, float* y, int64_t n); // <-- ADDED
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
//     const int n = 16;
//     std::vector<float> x(n), y(n);
//     for (int i = 0; i < n; ++i) x[i] = i / 4.0f - 2.0f;
//     auto start_your_gelu = std::chrono::high_resolution_clock::now();
//     gelu_impl_optimized(x.data(), y.data(), n);
//     auto end_your_gelu = std::chrono::high_resolution_clock::now();
//     std::cout << "GELU outputs:\n";
//     for (float v : y) std::cout << v << " ";
//     std::chrono::duration<double, std::milli> your_gelu_duration = (end_your_gelu - start_your_gelu) / num_relu_runs;
//     std::cout << "Our optimized gelu_impl average time: " << your_gelu_duration.count() << " ms" << std::endl;
//     std::cout << std::endl;

//     // Your optimized ReLU implementation
//     auto start_your_relu = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < num_relu_runs; ++i) {
//         // Call the new function name
//         relu_impl_optimized(x_relu.data(), y_relu.data(), n_relu); // <-- CHANGED
//     }
//     auto end_your_relu = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> your_relu_duration = (end_your_relu - start_your_relu) / num_relu_runs;
//     std::cout << "Our optimized relu_impl average time: " << your_relu_duration.count() << " ms" << std::endl;

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

//     // Your optimized MatMul implementation
//     auto start_your_matmul = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < num_matmul_runs; ++i) {
//         // Call the new function name
//         matmul_impl_optimized(A.data(), B.data(), C.data(), M, K, N); // <-- CHANGED
//     }
//     auto end_your_matmul = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> your_matmul_duration = (end_your_matmul - start_your_matmul) / num_matmul_runs;
//     std::cout << "Our optimized matmul_impl average time: " << your_matmul_duration.count() << " ms" << std::endl;

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

// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <random>
// #include <numeric>
// #include <Eigen/Dense>

// // -----------------------------------------------------------------------------
// // External optimized kernel declarations (from agkernels_cpu.cpp)
// // -----------------------------------------------------------------------------
// extern "C" {
//     void relu_impl_optimized(const float* x, float* y, int64_t n);
//     void leakyrelu_impl_optimized(const float* x, float* y, int64_t n, float alpha);
//     void gelu_impl_optimized(const float* x, float* y, int64_t n);
//     void matmul_impl_cudatile(const float* A, const float* B, float* C, int M, int K, int N);
//     void gemm_impl_optimized(const float* A, const float* B, const float* C, float* Out,
//                              int M, int K, int N);
// }

// // -----------------------------------------------------------------------------
// // Helper functions
// // -----------------------------------------------------------------------------
// void fill_random(std::vector<float>& vec, float low = -1.0f, float high = 1.0f) {
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<float> dis(low, high);
//     for (auto& v : vec) v = dis(gen);
// }

// template <typename F>
// double benchmark(F&& func, int runs = 50) {
//     auto start = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < runs; ++i) func();
//     auto end = std::chrono::high_resolution_clock::now();
//     return std::chrono::duration<double, std::milli>(end - start).count() / runs;
// }

// // -----------------------------------------------------------------------------
// // Main Benchmark
// // -----------------------------------------------------------------------------
// int main() {
//     const int64_t N = 10'000'000;  // large array for elementwise ops
//     const int M = 512, K = 512, P = 512; // for matmul

//     std::cout << "=============================\n";
//     std::cout << " CPU Optimized Kernel Tests  \n";
//     std::cout << "=============================\n\n";

//     // =======================================================================
//     // 1. ReLU
//     // =======================================================================
//     {
//         std::cout << "--- ReLU ---\n";
//         std::vector<float> x(N), y(N);
//         fill_random(x);

//         double t_our = benchmark([&]() { relu_impl_optimized(x.data(), y.data(), N); });
//         std::cout << "Our optimized ReLU avg time: " << t_our << " ms\n";

//         Eigen::Map<Eigen::VectorXf> x_e(x.data(), N);
//         Eigen::VectorXf y_e(N);
//         double t_eigen = benchmark([&]() { y_e = x_e.cwiseMax(0.f); });
//         std::cout << "Eigen ReLU avg time: " << t_eigen << " ms\n";
//         std::cout << "Speedup: " << t_eigen / t_our << "x\n\n";
//     }

//     // =======================================================================
//     // 2. LeakyReLU
//     // =======================================================================
//     {
//         std::cout << "--- LeakyReLU ---\n";
//         const float alpha = 0.01f;
//         std::vector<float> x(N), y(N);
//         fill_random(x);

//         double t_our = benchmark([&]() { leakyrelu_impl_optimized(x.data(), y.data(), N, alpha); });
//         std::cout << "Our optimized LeakyReLU avg time: " << t_our << " ms\n";

//         Eigen::Map<Eigen::VectorXf> x_e(x.data(), N);
//         Eigen::VectorXf y_e(N);
//         double t_eigen = benchmark([&]() {
//             y_e = x_e.unaryExpr([&](float v) { return v > 0.0f ? v : alpha * v; });
//         });
//         std::cout << "Eigen LeakyReLU avg time: " << t_eigen << " ms\n";
//         std::cout << "Speedup: " << t_eigen / t_our << "x\n\n";
//     }

//     // =======================================================================
//     // 3. GELU
//     // =======================================================================
//     {
//         std::cout << "--- GELU ---\n";
//         std::vector<float> x(N), y(N);
//         fill_random(x);

//         double t_our = benchmark([&]() { gelu_impl_optimized(x.data(), y.data(), N); });
//         std::cout << "Our optimized GELU avg time: " << t_our << " ms\n";

//         Eigen::Map<Eigen::VectorXf> x_e(x.data(), N);
//         Eigen::VectorXf y_e(N);
//         double t_eigen = benchmark([&]() {
//             y_e = x_e.unaryExpr([](float v) {
//                 float u = 0.7978845608f * (v + 0.044715f * v * v * v);
//                 return 0.5f * v * (1.0f + std::tanh(u));
//             });
//         });
//         std::cout << "Eigen GELU avg time: " << t_eigen << " ms\n";
//         std::cout << "Speedup: " << t_eigen / t_our << "x\n\n";
//     }

//     // =======================================================================
//     // 4. MatMul (C = A × B)
//     // =======================================================================
//     {
//         std::cout << "--- MatMul (512x512) ---\n";
//         std::vector<float> A(M * K), B(K * P), C(M * P);
//         fill_random(A);
//         fill_random(B);

//         double t_our = benchmark([&]() { matmul_impl_cudatile(A.data(), B.data(), C.data(), M, K, P); }, 10);
//         std::cout << "Our optimized MatMul avg time: " << t_our << " ms\n";

//         Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
//             A_e(A.data(), M, K), B_e(B.data(), K, P);
//         Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> C_e(M, P);

//         double t_eigen = benchmark([&]() { C_e.noalias() = A_e * B_e; }, 10);
//         std::cout << "Eigen MatMul avg time: " << t_eigen << " ms\n";
//         std::cout << "Speedup: " << t_eigen / t_our << "x\n\n";
//     }

//     // =======================================================================
//     // 5. FMA / GEMM (Out = A×B + C)
//     // =======================================================================
//     {
//         std::cout << "--- FMA (A×B + C) ---\n";
//         std::vector<float> A(M * K), B(K * P), C(M * P), Out(M * P);
//         fill_random(A);
//         fill_random(B);
//         fill_random(C);

//         double t_our = benchmark([&]() {
//             gemm_impl_optimized(A.data(), B.data(), C.data(), Out.data(), M, K, P);
//         }, 10);
//         std::cout << "Our optimized FMA avg time: " << t_our << " ms\n";

//         Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
//             A_e(A.data(), M, K), B_e(B.data(), K, P), C_e(C.data(), M, P);
//         Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Out_e(M, P);

//         double t_eigen = benchmark([&]() {
//             Out_e.noalias() = A_e * B_e + C_e;
//         }, 10);
//         std::cout << "Eigen FMA avg time: " << t_eigen << " ms\n";
//         std::cout << "Speedup: " << t_eigen / t_our << "x\n\n";
//     }

//     std::cout << "=============================\n";
//     std::cout << " Benchmark Complete ✅\n";
//     std::cout << "=============================\n";
//     return 0;
// }


// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <random>
// #include <cmath>
// #include <Eigen/Dense>

// // -----------------------------------------------------------------------------
// // Declare your optimized sigmoid kernel
// // -----------------------------------------------------------------------------
// extern "C" {
//     void sigmoid_impl_optimized(const float* x, float* y, int64_t n);
// }

// // -----------------------------------------------------------------------------
// // Helper functions
// // -----------------------------------------------------------------------------
// void fill_random(std::vector<float>& vec, float low = -8.0f, float high = 8.0f) {
//     std::mt19937 gen(std::random_device{}());
//     std::uniform_real_distribution<float> dis(low, high);
//     for (auto& v : vec) v = dis(gen);
// }

// template <typename F>
// double benchmark(F&& func, int runs = 50) {
//     auto start = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < runs; ++i) func();
//     auto end = std::chrono::high_resolution_clock::now();
//     return std::chrono::duration<double, std::milli>(end - start).count() / runs;
// }

// float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
//     float max_diff = 0.0f;
//     for (size_t i = 0; i < a.size(); ++i) {
//         float diff = std::fabs(a[i] - b[i]);
//         if (diff > max_diff) max_diff = diff;
//     }
//     return max_diff;
// }

// // -----------------------------------------------------------------------------
// // Main Test + Benchmark
// // -----------------------------------------------------------------------------
// int main() {
//     const int64_t N = 10'000'000;  // 10 million elements
//     const int num_runs = 100;

//     std::cout << "=============================\n";
//     std::cout << "   Sigmoid Optimized Kernel  \n";
//     std::cout << "=============================\n\n";

//     // --- Correctness test on small input ---
//     {
//         std::cout << "--- Sigmoid Correctness ---\n";
//         const int n_small = 16;
//         std::vector<float> x_small(n_small), y_small(n_small);
//         for (int i = 0; i < n_small; ++i) x_small[i] = i / 4.0f - 2.0f;

//         sigmoid_impl_optimized(x_small.data(), y_small.data(), n_small);

//         std::cout << "Input:  ";
//         for (float v : x_small) std::cout << v << " ";
//         std::cout << "\nOutput: ";
//         for (float v : y_small) std::cout << v << " ";
//         std::cout << "\n\n";
//     }

//     // --- Benchmark setup ---
//     std::cout << "--- Sigmoid Benchmark (" << N << " elements, "
//               << num_runs << " runs) ---\n";
//     std::vector<float> x(N), y(N), y_ref(N);
//     fill_random(x);

//     // --- Our optimized Sigmoid ---
//     double t_our = benchmark([&]() { sigmoid_impl_optimized(x.data(), y.data(), N); }, num_runs);

//     // --- Eigen reference ---
//     Eigen::Map<Eigen::VectorXf> x_eigen(x.data(), N);
//     Eigen::VectorXf y_eigen(N);
//     double t_eigen = benchmark([&]() {
//         y_eigen = x_eigen.unaryExpr([](float v) { return 1.0f / (1.0f + std::exp(-v)); });
//     }, num_runs);

//     // --- Compute error ---
//     std::memcpy(y_ref.data(), y_eigen.data(), N * sizeof(float));
//     float max_err = max_abs_diff(y, y_ref);

//     // --- Print results ---
//     std::cout << "✅ Our optimized sigmoid_impl:\n";
//     std::cout << "   Average time per run = " << t_our << " ms\n";
//     std::cout << "🧩 Eigen reference:\n";
//     std::cout << "   Average time per run = " << t_eigen << " ms\n";
//     std::cout << "🚀 Speedup: " << (t_eigen / t_our) << "x faster\n";
//     std::cout << "📊 Max absolute error: " << max_err << "\n";

//     std::cout << "\nBenchmark Complete ✅\n";
//     std::cout << "=============================\n";

//     return 0;
// }

// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <random>
// #include <cmath>
// #include <Eigen/Dense>

// // -----------------------------------------------------------------------------
// // Declare your optimized Softplus kernel
// // -----------------------------------------------------------------------------
// extern "C" {
//     void softplus_impl_optimized(const float* x, float* y, int64_t n);
// }

// // -----------------------------------------------------------------------------
// // Helper functions
// // -----------------------------------------------------------------------------
// void fill_random(std::vector<float>& vec, float low = -8.0f, float high = 8.0f) {
//     std::mt19937 gen(std::random_device{}());
//     std::uniform_real_distribution<float> dis(low, high);
//     for (auto& v : vec) v = dis(gen);
// }

// template <typename F>
// double benchmark(F&& func, int runs = 50) {
//     auto start = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < runs; ++i) func();
//     auto end = std::chrono::high_resolution_clock::now();
//     return std::chrono::duration<double, std::milli>(end - start).count() / runs;
// }

// float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
//     float max_diff = 0.0f;
//     for (size_t i = 0; i < a.size(); ++i) {
//         float diff = std::fabs(a[i] - b[i]);
//         if (diff > max_diff) max_diff = diff;
//     }
//     return max_diff;
// }

// // -----------------------------------------------------------------------------
// // Main Test + Benchmark
// // -----------------------------------------------------------------------------
// int main() {
//     const int64_t N = 10'000'000;  // 10 million elements
//     const int num_runs = 100;

//     std::cout << "=============================\n";
//     std::cout << "  Softplus Optimized Kernel  \n";
//     std::cout << "=============================\n\n";

//     // --- Correctness test on small input ---
//     {
//         std::cout << "--- Softplus Correctness ---\n";
//         const int n_small = 16;
//         std::vector<float> x_small(n_small), y_small(n_small);
//         for (int i = 0; i < n_small; ++i) x_small[i] = i / 2.0f - 4.0f; // range [-4,4]

//         softplus_impl_optimized(x_small.data(), y_small.data(), n_small);

//         std::cout << "Input:  ";
//         for (float v : x_small) std::cout << v << " ";
//         std::cout << "\nOutput: ";
//         for (float v : y_small) std::cout << v << " ";
//         std::cout << "\n\n";
//     }

//     // --- Benchmark setup ---
//     std::cout << "--- Softplus Benchmark (" << N << " elements, "
//               << num_runs << " runs) ---\n";
//     std::vector<float> x(N), y(N), y_ref(N);
//     fill_random(x);

//     // --- Our optimized Softplus ---
//     double t_our = benchmark([&]() { softplus_impl_optimized(x.data(), y.data(), N); }, num_runs);

//     // --- Eigen reference ---
//     Eigen::Map<Eigen::VectorXf> x_eigen(x.data(), N);
//     Eigen::VectorXf y_eigen(N);
//     double t_eigen = benchmark([&]() {
//         y_eigen = x_eigen.unaryExpr([](float v) {
//             // Stable softplus
//             return (v > 0.0f) ? v + std::log1pf(std::exp(-v)) : std::log1pf(std::exp(v));
//         });
//     }, num_runs);

//     // --- Compute accuracy ---
//     std::memcpy(y_ref.data(), y_eigen.data(), N * sizeof(float));
//     float max_err = max_abs_diff(y, y_ref);

//     // --- Print results ---
//     std::cout << "✅ Our optimized softplus_impl:\n";
//     std::cout << "   Average time per run = " << t_our << " ms\n";
//     std::cout << "🧩 Eigen reference:\n";
//     std::cout << "   Average time per run = " << t_eigen << " ms\n";
//     std::cout << "🚀 Speedup: " << (t_eigen / t_our) << "x faster\n";
//     std::cout << "📊 Max absolute error: " << max_err << "\n";

//     std::cout << "\nBenchmark Complete ✅\n";
//     std::cout << "=============================\n";

//     return 0;
// }

// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <random>
// #include <cmath>
// #include <Eigen/Dense>

// // -----------------------------------------------------------------------------
// // Declare your optimized kernels
// // -----------------------------------------------------------------------------
// extern "C" {
//     void exp_impl_optimized(const float* x, float* y, int64_t n);
//     void log_impl_optimized(const float* x, float* y, int64_t n);
// }

// // -----------------------------------------------------------------------------
// // Helper utilities
// // -----------------------------------------------------------------------------
// void fill_random(std::vector<float>& vec, float low, float high) {
//     std::mt19937 gen(std::random_device{}());
//     std::uniform_real_distribution<float> dis(low, high);
//     for (auto& v : vec) v = dis(gen);
// }

// template <typename Fn>
// double benchmark(Fn&& func, int runs = 50) {
//     auto start = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < runs; ++i)
//         func();
//     auto end = std::chrono::high_resolution_clock::now();
//     return std::chrono::duration<double, std::milli>(end - start).count() / runs;
// }

// float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
//     float maxd = 0.0f;
//     for (size_t i = 0; i < a.size(); ++i)
//         maxd = std::max(maxd, std::fabs(a[i] - b[i]));
//     return maxd;
// }

// // -----------------------------------------------------------------------------
// // Main Benchmark
// // -----------------------------------------------------------------------------
// int main() {
//     const int64_t N = 10'000'000;
//     const int num_runs = 100;

//     std::cout << "=====================================\n";
//     std::cout << "   Optimized exp/log Kernel Test     \n";
//     std::cout << "=====================================\n\n";

//     // =======================================================================
//     // 1. Correctness: exp_impl_optimized
//     // =======================================================================
//     {
//         std::cout << "--- exp_impl_optimized correctness ---\n";
//         const int n_small = 16;
//         std::vector<float> x_small(n_small), y_small(n_small);
//         for (int i = 0; i < n_small; ++i) x_small[i] = (i - 8) / 2.0f; // range [-4,4]

//         exp_impl_optimized(x_small.data(), y_small.data(), n_small);

//         std::cout << "Input:  ";
//         for (float v : x_small) std::cout << v << " ";
//         std::cout << "\nOutput: ";
//         for (float v : y_small) std::cout << v << " ";
//         std::cout << "\n\n";
//     }

//     // =======================================================================
//     // 2. Correctness: log_impl_optimized
//     // =======================================================================
//     {
//         std::cout << "--- log_impl_optimized correctness ---\n";
//         const int n_small = 16;
//         std::vector<float> x_small(n_small), y_small(n_small);
//         for (int i = 0; i < n_small; ++i) x_small[i] = (i + 1) * 0.5f; // range [0.5,8.0]

//         log_impl_optimized(x_small.data(), y_small.data(), n_small);

//         std::cout << "Input:  ";
//         for (float v : x_small) std::cout << v << " ";
//         std::cout << "\nOutput: ";
//         for (float v : y_small) std::cout << v << " ";
//         std::cout << "\n\n";
//     }

//     // =======================================================================
//     // 3. Benchmark exp
//     // =======================================================================
//     std::cout << "--- Benchmark: exp_impl_optimized ---\n";
//     std::vector<float> x_exp(N), y_exp(N), y_exp_ref(N);
//     fill_random(x_exp, -10.0f, 10.0f);

//     // Our kernel
//     double t_our_exp = benchmark([&]() { exp_impl_optimized(x_exp.data(), y_exp.data(), N); }, num_runs);

//     // Eigen reference
//     Eigen::Map<Eigen::VectorXf> x_e_exp(x_exp.data(), N);
//     Eigen::VectorXf y_e_exp(N);
//     double t_eigen_exp = benchmark([&]() { y_e_exp = x_e_exp.array().exp(); }, num_runs);

//     std::memcpy(y_exp_ref.data(), y_e_exp.data(), N * sizeof(float));
//     float err_exp = max_abs_diff(y_exp, y_exp_ref);

//     std::cout << "✅ Optimized exp_impl_optimized avg time: " << t_our_exp << " ms\n";
//     std::cout << "🧩 Eigen exp reference avg time:          " << t_eigen_exp << " ms\n";
//     std::cout << "🚀 Speedup: " << (t_eigen_exp / t_our_exp) << "x\n";
//     std::cout << "📊 Max abs error: " << err_exp << "\n\n";

//     // =======================================================================
//     // 4. Benchmark log
//     // =======================================================================
//     std::cout << "--- Benchmark: log_impl_optimized ---\n";
//     std::vector<float> x_log(N), y_log(N), y_log_ref(N);
//     fill_random(x_log, 0.001f, 100.0f);  // log(x) defined for x>0

//     // Our kernel
//     double t_our_log = benchmark([&]() { log_impl_optimized(x_log.data(), y_log.data(), N); }, num_runs);

//     // Eigen reference
//     Eigen::Map<Eigen::VectorXf> x_e_log(x_log.data(), N);
//     Eigen::VectorXf y_e_log(N);
//     double t_eigen_log = benchmark([&]() { y_e_log = x_e_log.array().log(); }, num_runs);

//     std::memcpy(y_log_ref.data(), y_e_log.data(), N * sizeof(float));
//     float err_log = max_abs_diff(y_log, y_log_ref);

//     std::cout << "✅ Optimized log_impl_optimized avg time: " << t_our_log << " ms\n";
//     std::cout << "🧩 Eigen log reference avg time:          " << t_eigen_log << " ms\n";
//     std::cout << "🚀 Speedup: " << (t_eigen_log / t_our_log) << "x\n";
//     std::cout << "📊 Max abs error: " << err_log << "\n\n";

//     std::cout << "=====================================\n";
//     std::cout << " Benchmark Complete ✅\n";
//     std::cout << "=====================================\n";
//     return 0;
// }

// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <random>
// #include <cmath>
// #include <Eigen/Dense>

// // -----------------------------------------------------------------------------
// // Declare your optimized kernels
// // -----------------------------------------------------------------------------
// extern "C" {
//     void sqrt_impl_optimized(const float* x, float* y, int64_t n);
//     void pow_impl_optimized(const float* x, float* y, int64_t n, float exponent);
// }

// // -----------------------------------------------------------------------------
// // Helper functions
// // -----------------------------------------------------------------------------
// void fill_random(std::vector<float>& vec, float low, float high) {
//     std::mt19937 gen(std::random_device{}());
//     std::uniform_real_distribution<float> dis(low, high);
//     for (auto& v : vec) v = dis(gen);
// }

// template <typename Fn>
// double benchmark(Fn&& func, int runs = 50) {
//     auto start = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < runs; ++i) func();
//     auto end = std::chrono::high_resolution_clock::now();
//     return std::chrono::duration<double, std::milli>(end - start).count() / runs;
// }

// float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
//     float maxd = 0.0f;
//     for (size_t i = 0; i < a.size(); ++i)
//         maxd = std::max(maxd, std::fabs(a[i] - b[i]));
//     return maxd;
// }

// // -----------------------------------------------------------------------------
// // Main Test + Benchmark
// // -----------------------------------------------------------------------------
// int main() {
//     const int64_t N = 10'000'000;
//     const int num_runs = 100;

//     std::cout << "=====================================\n";
//     std::cout << "   Optimized sqrt & pow Kernels      \n";
//     std::cout << "=====================================\n\n";

//     // =======================================================================
//     // 1. Correctness Test: sqrt_impl_optimized
//     // =======================================================================
//     {
//         std::cout << "--- sqrt_impl_optimized correctness ---\n";
//         const int n_small = 16;
//         std::vector<float> x_small(n_small), y_small(n_small);
//         for (int i = 0; i < n_small; ++i) x_small[i] = static_cast<float>(i);

//         sqrt_impl_optimized(x_small.data(), y_small.data(), n_small);

//         std::cout << "Input:  ";
//         for (float v : x_small) std::cout << v << " ";
//         std::cout << "\nOutput: ";
//         for (float v : y_small) std::cout << v << " ";
//         std::cout << "\n\n";
//     }

//     // =======================================================================
//     // 2. Correctness Test: pow_impl_optimized
//     // =======================================================================
//     {
//         std::cout << "--- pow_impl_optimized correctness ---\n";
//         const int n_small = 16;
//         std::vector<float> x_small(n_small), y_small(n_small);
//         for (int i = 0; i < n_small; ++i) x_small[i] = (i + 1) * 0.5f; // [0.5, 8]
//         float exponent = 1.5f;

//         pow_impl_optimized(x_small.data(), y_small.data(), n_small, exponent);

//         std::cout << "Input:  ";
//         for (float v : x_small) std::cout << v << " ";
//         std::cout << "\nOutput (x^1.5): ";
//         for (float v : y_small) std::cout << v << " ";
//         std::cout << "\n\n";
//     }

//     // =======================================================================
//     // 3. Benchmark sqrt
//     // =======================================================================
//     std::cout << "--- Benchmark: sqrt_impl_optimized ---\n";
//     std::vector<float> x_sqrt(N), y_sqrt(N), y_sqrt_ref(N);
//     fill_random(x_sqrt, 0.0f, 100.0f);

//     // Our kernel
//     double t_our_sqrt = benchmark([&]() { sqrt_impl_optimized(x_sqrt.data(), y_sqrt.data(), N); }, num_runs);

//     // Eigen reference
//     Eigen::Map<Eigen::VectorXf> x_e_sqrt(x_sqrt.data(), N);
//     Eigen::VectorXf y_e_sqrt(N);
//     double t_eigen_sqrt = benchmark([&]() { y_e_sqrt = x_e_sqrt.array().sqrt(); }, num_runs);

//     std::memcpy(y_sqrt_ref.data(), y_e_sqrt.data(), N * sizeof(float));
//     float err_sqrt = max_abs_diff(y_sqrt, y_sqrt_ref);

//     std::cout << "✅ Optimized sqrt_impl_optimized avg time: " << t_our_sqrt << " ms\n";
//     std::cout << "🧩 Eigen sqrt reference avg time:          " << t_eigen_sqrt << " ms\n";
//     std::cout << "🚀 Speedup: " << (t_eigen_sqrt / t_our_sqrt) << "x\n";
//     std::cout << "📊 Max abs error: " << err_sqrt << "\n\n";

//     // =======================================================================
//     // 4. Benchmark pow
//     // =======================================================================
//     std::cout << "--- Benchmark: pow_impl_optimized ---\n";
//     std::vector<float> x_pow(N), y_pow(N), y_pow_ref(N);
//     fill_random(x_pow, 0.001f, 100.0f);
//     float exponent = 1.75f;

//     // Our kernel
//     double t_our_pow = benchmark([&]() { pow_impl_optimized(x_pow.data(), y_pow.data(), N, exponent); }, num_runs);

//     // Eigen reference
//     Eigen::Map<Eigen::VectorXf> x_e_pow(x_pow.data(), N);
//     Eigen::VectorXf y_e_pow(N);
//     double t_eigen_pow = benchmark([&]() { y_e_pow = x_e_pow.array().pow(exponent); }, num_runs);

//     std::memcpy(y_pow_ref.data(), y_e_pow.data(), N * sizeof(float));
//     float err_pow = max_abs_diff(y_pow, y_pow_ref);

//     std::cout << "✅ Optimized pow_impl_optimized avg time: " << t_our_pow << " ms\n";
//     std::cout << "🧩 Eigen pow reference avg time:          " << t_eigen_pow << " ms\n";
//     std::cout << "🚀 Speedup: " << (t_eigen_pow / t_our_pow) << "x\n";
//     std::cout << "📊 Max abs error: " << err_pow << "\n\n";

//     std::cout << "=====================================\n";
//     std::cout << " Benchmark Complete ✅\n";
//     std::cout << "=====================================\n";
//     return 0;
// }

// test_relu_bwd.cpp (main snippet)
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
extern "C" {
    void relu_bwd_impl_optimized(const float* x, const float* dY, float* dX, int64_t n);
}

void relu_bwd_naive(const float* x, const float* dY, float* dX, int64_t n) {
    for (int64_t i = 0; i < n; ++i) dX[i] = x[i] > 0.0f ? dY[i] : 0.0f;
}

int main(){
    const int64_t N = 10'000'000;
    std::vector<float> x(N), dy(N), dx_naive(N), dx_opt(N);
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dis(-3.0f, 3.0f);
    for (auto& v: x) v = dis(gen);
    for (auto& v: dy) v = dis(gen);

    // warmup
    relu_bwd_impl_optimized(x.data(), dy.data(), dx_opt.data(), N);

    // time optimized
    auto t0 = std::chrono::high_resolution_clock::now();
    relu_bwd_impl_optimized(x.data(), dy.data(), dx_opt.data(), N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double topt = std::chrono::duration<double, std::milli>(t1-t0).count();

    // naive
    auto t2 = std::chrono::high_resolution_clock::now();
    relu_bwd_naive(x.data(), dy.data(), dx_naive.data(), N);
    auto t3 = std::chrono::high_resolution_clock::now();
    double tnaive = std::chrono::duration<double, std::milli>(t3-t2).count();

    // check errors
    float max_err = 0.f; double mse=0.0;
    for (int64_t i=0;i<N;++i) {
        float d = dx_naive[i] - dx_opt[i];
        max_err = std::max(max_err, std::fabs(d));
        mse += double(d)*double(d);
    }
    mse = std::sqrt(mse / double(N));
    std::cout<<"ReLU bwd: naive="<<tnaive<<" ms, opt="<<topt<<" ms, max_err="<<max_err<<", rmse="<<mse<<"\n";
}


// ============================================================
// File: test_checkpoint_cpu_kernels.cpp
// Purpose: Verify gradient checkpointing + optimized CPU kernels
// ============================================================

// #include <iostream>
// #include <vector>
// #include <cmath>
// #include "ad/ag_all.hpp"
// #include "ad/checkpoint.hpp"
// #include "ad/kernels_api.hpp"
// #include "tensor.hpp"
// #include "ad/ops.hpp"

// using namespace ag;

// static bool allclose(const Tensor& A, const Tensor& B, float tol = 1e-5f) {
//     if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
//     const float* a = A.data();
//     const float* b = B.data();
//     for (int i = 0; i < A.size(); ++i) {
//         float diff = std::fabs(a[i] - b[i]);
//         if (diff > tol) return false;
//     }
//     return true;
// }


// int main() {
//     std::cout << "===== Gradient Checkpointing + CPU Kernel Test =====\n";

//     // ------------------------------------------------------------
//     // 1. Load optimized CPU kernels
//     // ------------------------------------------------------------
//     std::cout << "Loading optimized CPU kernels...\n";
//     ag::kernels::load_cpu_plugin("./libagkernels_cpu.so");
//     std::cout << "✅ CPU kernels loaded successfully\n";

//     // ------------------------------------------------------------
//     // 2. Prepare small deterministic tensors
//     // ------------------------------------------------------------
//     Tensor x_data = Tensor::randn(4, 4, 42);
//     Tensor W1_data = Tensor::randn(4, 4, 123);
//     Tensor W2_data = Tensor::randn(4, 4, 999);
//     Tensor b1_data = Tensor::randn(1, 4, 7);
//     Tensor b2_data = Tensor::randn(1, 4, 11);

//     // ------------------------------------------------------------
//     // 3. Wrap them as Values for the computational graph
//     // ------------------------------------------------------------
//     Value x = constant(x_data, "x");
//     Value W1 = param(W1_data, "W1");
//     Value W2 = param(W2_data, "W2");
//     Value b1 = param(b1_data, "b1");
//     Value b2 = param(b2_data, "b2");

//     // ------------------------------------------------------------
//     // 4. Build a two-layer MLP with checkpointed hidden layer
//     // ------------------------------------------------------------
//     Value h1 = relu(add(matmul(x, W1), b1));
//     // mark h1 as a recomputation checkpoint
//     h1 = inplace_checkpoint(h1);

//     Value y = relu(add(matmul(h1, W2), b2));
//     Value loss = sum(y);

//     std::cout << "\nForward complete. Starting backward...\n";

//     // ------------------------------------------------------------
//     // 5. Backward pass (optimized kernels are used)
//     // ------------------------------------------------------------
//     zero_grad(loss);
//     backward(loss);

//     std::cout << "Backward complete ✅\n";

//     // ------------------------------------------------------------
//     // 6. Verify checkpoint metadata
//     // ------------------------------------------------------------
//     auto n = h1.node;
//     std::cout << "\n--- Checkpoint verification ---\n";
//     if (n->is_checkpoint)
//         std::cout << "Node " << n->debug_name << " is checkpointed ✅\n";
//     else
//         std::cout << "Node " << n->debug_name << " is NOT checkpointed ❌\n";

//     // ------------------------------------------------------------
//     // 7. Print parameter gradients
//     // ------------------------------------------------------------
//     std::cout << "\nGradients (with checkpointing):\n";
//     std::cout << "dL/dW1:\n" << W1.grad();
//     std::cout << "dL/dW2:\n" << W2.grad();
//     std::cout << "dL/db1:\n" << b1.grad();
//     std::cout << "dL/db2:\n" << b2.grad();

//     // ------------------------------------------------------------
//     // 8. Recompute checkpointed subgraph manually
//     // ------------------------------------------------------------
//     std::cout << "\n--- Manual recomputation test ---\n";
//     bool ok = checkpoint_impl::recompute_subgraph(h1.node->shared_from_this());
//     std::cout << (ok ? "Recomputation success ✅" : "Recomputation failed ❌") << "\n";
//     std::cout << "Recomputed value for checkpointed node:\n" << h1.node->value << "\n";

//     // ------------------------------------------------------------
//     // 9. Compare with non-checkpointed forward/backward
//     // ------------------------------------------------------------
//     std::cout << "\nRunning non-checkpointed baseline for comparison...\n";

//     // rebuild the same network without checkpoint
//     Value x2 = constant(x_data, "x2");
//     Value W1b = param(Tensor(W1_data), "W1b");
//     Value W2b = param(Tensor(W2_data), "W2b");
//     Value b1b = param(Tensor(b1_data), "b1b");
//     Value b2b = param(Tensor(b2_data), "b2b");
//     Value h1b = relu(add(matmul(x2, W1b), b1b));
//     Value yb  = relu(add(matmul(h1b, W2b), b2b));
//     Value loss_b = sum(yb);

//     zero_grad(loss_b);
//     backward(loss_b);

//     std::cout << "Comparing gradients...\n";
//     bool grad_ok = allclose(W1.grad(), W1b.grad()) &&
//                    allclose(W2.grad(), W2b.grad()) &&
//                    allclose(b1.grad(), b1b.grad()) &&
//                    allclose(b2.grad(), b2b.grad());

//     if (grad_ok)
//         std::cout << "✅ Gradients match between checkpointed and non-checkpointed runs.\n";
//     else
//         std::cout << "❌ Gradient mismatch detected!\n";

//     // ------------------------------------------------------------
//     // 10. Summary
//     // ------------------------------------------------------------
//     std::cout << "\n===== Checkpoint + Optimized CPU Kernel Test Completed =====\n";
//     return 0;
// }
