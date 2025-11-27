#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

int main() {
    const int ITERATIONS = 50;
    const int WARMUP = 5;

    cout << "=========================================\n";
    cout << "   Final Performance Comparison\n";
    cout << "   512x512 Matrix Multiplication\n";
    cout << "=========================================\n\n";

    // Benchmark MLIR Sequential
    cout << "1. MLIR Sequential (Vectorized)\n";
    cout << "   Warming up...\n";

    for (int i = 0; i < WARMUP; i++) {
        system("./matmul > /dev/null 2>&1");
    }

    cout << "   Running " << ITERATIONS << " iterations...\n";
    vector<double> times_mlir;

    for (int i = 0; i < ITERATIONS; i++) {
        auto start = high_resolution_clock::now();
        int ret = system("./matmul > /dev/null 2>&1");
        auto end = high_resolution_clock::now();

        if (ret != 0) {
            cerr << "   ERROR: ./matmul failed\n";
            return 1;
        }

        duration<double> elapsed = end - start;
        times_mlir.push_back(elapsed.count() * 1000); // Convert to ms
    }

    sort(times_mlir.begin(), times_mlir.end());

    // Use median and best times to avoid outliers
    double median_mlir = times_mlir[ITERATIONS / 2];
    double best_mlir = times_mlir[0];
    double worst_mlir = times_mlir[ITERATIONS - 1];

    double avg_mlir = 0;
    for (double t : times_mlir) avg_mlir += t;
    avg_mlir /= ITERATIONS;

    double gflops_best = (2.0 * 512 * 512 * 512) / (best_mlir / 1000.0 * 1e9);
    double gflops_median = (2.0 * 512 * 512 * 512) / (median_mlir / 1000.0 * 1e9);

    printf("   Best:    %.3f ms (%.2f GFLOPS)\n", best_mlir, gflops_best);
    printf("   Median:  %.3f ms (%.2f GFLOPS)\n", median_mlir, gflops_median);
    printf("   Average: %.3f ms\n", avg_mlir);
    printf("   Worst:   %.3f ms\n\n", worst_mlir);

    // Benchmark OpenMP Parallel
    setenv("OMP_NUM_THREADS", "16", 1);
    cout << "2. C++ OpenMP Parallel (16 threads)\n";
    cout << "   Warming up...\n";

    for (int i = 0; i < WARMUP; i++) {
        system("./matmul_omp > /dev/null 2>&1");
    }

    cout << "   Running " << ITERATIONS << " iterations...\n";
    vector<double> times_omp;

    for (int i = 0; i < ITERATIONS; i++) {
        auto start = high_resolution_clock::now();
        int ret = system("./matmul_omp > /dev/null 2>&1");
        auto end = high_resolution_clock::now();

        if (ret != 0) {
            cerr << "   ERROR: ./matmul_omp failed\n";
            return 1;
        }

        duration<double> elapsed = end - start;
        times_omp.push_back(elapsed.count() * 1000); // Convert to ms
    }

    sort(times_omp.begin(), times_omp.end());

    double median_omp = times_omp[ITERATIONS / 2];
    double best_omp = times_omp[0];
    double worst_omp = times_omp[ITERATIONS - 1];

    double avg_omp = 0;
    for (double t : times_omp) avg_omp += t;
    avg_omp /= ITERATIONS;

    gflops_best = (2.0 * 512 * 512 * 512) / (best_omp / 1000.0 * 1e9);
    gflops_median = (2.0 * 512 * 512 * 512) / (median_omp / 1000.0 * 1e9);

    printf("   Best:    %.3f ms (%.2f GFLOPS)\n", best_omp, gflops_best);
    printf("   Median:  %.3f ms (%.2f GFLOPS)\n", median_omp, gflops_median);
    printf("   Average: %.3f ms\n", avg_omp);
    printf("   Worst:   %.3f ms\n\n", worst_omp);

    cout << "=========================================\n";
    printf("   Speedup (median): %.2fx\n", median_mlir / median_omp);
    printf("   Speedup (best):   %.2fx\n", best_mlir / best_omp);
    cout << "=========================================\n";

    return 0;
}
