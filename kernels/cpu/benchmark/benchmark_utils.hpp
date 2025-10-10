#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <iomanip>
#include <functional>

// Function to fill a vector with random floats
void fill_random(std::vector<float>& vec) {
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& v : vec) {
        v = dis(gen);
    }
}

// Simple Timer class
class Timer {
public:
    void start() {
        m_start = std::chrono::high_resolution_clock::now();
    }
    double stop() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - m_start).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

// GFLOPS calculation
double calculate_gflops(int M, int K, int N, double ms) {
    if (ms == 0) return 0.0;
    // Each output element requires K multiplications and K-1 additions, approx 2*K ops
    double ops = 2.0 * M * N * K;
    double seconds = ms / 1000.0;
    return (ops / seconds) / 1e9;
}

// The core benchmark runner
void run_matmul_benchmark(const std::string& name,
                         std::function<void(const float*, const float*, float*, int, int, int)> matmul_func,
                         const std::vector<float>& A,
                         const std::vector<float>& B,
                         std::vector<float>& C,
                         int M, int K, int N,
                         int runs) {
    Timer timer;
    double total_ms = 0;

    // Warm-up run
    matmul_func(A.data(), B.data(), C.data(), M, K, N);

    // Timed runs
    for (int i = 0; i < runs; ++i) {
        timer.start();
        matmul_func(A.data(), B.data(), C.data(), M, K, N);
        total_ms += timer.stop();
    }

    double avg_ms = total_ms / runs;
    double gflops = calculate_gflops(M, K, N, avg_ms);

    std::cout << std::left << std::setw(12) << name
              << ": " << std::fixed << std::setprecision(3) << std::setw(10) << avg_ms << " ms"
              << " | " << std::fixed << std::setprecision(2) << std::setw(8) << gflops << " GFLOPS" << std::endl;
}