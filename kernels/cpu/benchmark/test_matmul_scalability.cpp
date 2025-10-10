#include "benchmark_utils.hpp"
#include <omp.h> // For omp_set_num_threads

extern "C" {
    void matmul_impl_optimized(const float*, const float*, float*, int, int, int);
}

int main() {
    std::cout << "===== MatMul Thread Scalability Benchmark =====" << std::endl;
    const int M = 1024, K = 1024, N = 1024;
    const int runs = 5;
    std::cout << "--- Size: " << M << "x" << K << "x" << N << " (" << runs << " runs) ---\n" << std::endl;

    std::vector<float> A(M * K), B(K * N), C(M * N);
    fill_random(A);
    fill_random(B);

    double baseline_ms = 0.0;
    int max_threads = omp_get_max_threads();

    std::cout << std::setw(10) << "Threads" << std::setw(15) << "Time (ms)"
              << std::setw(15) << "GFLOPS" << std::setw(10) << "Speedup" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    for (int t = 1; t <= max_threads; t *= 2) {
        omp_set_num_threads(t);
        Timer timer;
        double total_ms = 0;
        matmul_impl_optimized(A.data(), B.data(), C.data(), M, K, N); // Warm-up
        for (int i = 0; i < runs; ++i) {
            timer.start();
            matmul_impl_optimized(A.data(), B.data(), C.data(), M, K, N);
            total_ms += timer.stop();
        }
        double avg_ms = total_ms / runs;

        if (t == 1) {
            baseline_ms = avg_ms;
        }

        std::cout << std::setw(10) << t
                  << std::fixed << std::setprecision(3) << std::setw(15) << avg_ms
                  << std::fixed << std::setprecision(2) << std::setw(15) << calculate_gflops(M, K, N, avg_ms)
                  << std::fixed << std::setprecision(2) << std::setw(10) << (baseline_ms / avg_ms) << "x"
                  << std::endl;
    }
    return 0;
}