// Internal benchmark - no process overhead
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>
#include <omp.h>

using namespace std;
using namespace std::chrono;

// Sequential matmul (like MLIR generates)
void matmul_sequential(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    const int N = 512;

    // Initialize C
    memset(C, 0, N * N * sizeof(float));

    // Tiled matmul with manual optimization
    const int TILE = 32;
    for (int i0 = 0; i0 < N; i0 += TILE) {
        for (int j0 = 0; j0 < N; j0 += TILE) {
            for (int k0 = 0; k0 < N; k0 += TILE) {
                int i_end = min(i0 + TILE, N);
                int j_end = min(j0 + TILE, N);
                int k_end = min(k0 + TILE, N);

                for (int i = i0; i < i_end; i++) {
                    for (int k = k0; k < k_end; k++) {
                        float a_ik = A[i * N + k];
                        #pragma GCC ivdep
                        for (int j = j0; j < j_end; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

// Parallel OpenMP matmul
void matmul_parallel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    const int N = 512;
    const int TILE = 64;

    // Initialize C
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0f;
        }
    }

    // Parallel tiled matmul
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i0 = 0; i0 < N; i0 += TILE) {
        for (int j0 = 0; j0 < N; j0 += TILE) {
            for (int k0 = 0; k0 < N; k0 += TILE) {
                int i_end = min(i0 + TILE, N);
                int j_end = min(j0 + TILE, N);
                int k_end = min(k0 + TILE, N);

                for (int i = i0; i < i_end; i++) {
                    for (int k = k0; k < k_end; k++) {
                        float a_ik = A[i * N + k];
                        #pragma omp simd
                        for (int j = j0; j < j_end; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

int main() {
    const int N = 512;
    const int SIZE = N * N;
    const int ITERATIONS = 30;
    const int WARMUP = 3;

    // Allocate aligned memory
    float *A = (float*)aligned_alloc(64, SIZE * sizeof(float));
    float *B = (float*)aligned_alloc(64, SIZE * sizeof(float));
    float *C = (float*)aligned_alloc(64, SIZE * sizeof(float));

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize
    for (int i = 0; i < SIZE; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    cout << "=========================================\n";
    cout << "   Pure Computation Benchmark (No Process Overhead)\n";
    cout << "   512x512 matrices\n";
    cout << "=========================================\n\n";

    // Benchmark Sequential
    cout << "1. Sequential (Tiled + Vectorized)\n";

    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        matmul_sequential(A, B, C);
    }

    vector<double> times_seq;
    for (int i = 0; i < ITERATIONS; i++) {
        auto start = high_resolution_clock::now();
        matmul_sequential(A, B, C);
        auto end = high_resolution_clock::now();

        duration<double> elapsed = end - start;
        times_seq.push_back(elapsed.count());
    }

    // Verify
    if (C[0] != 1024.0f) {
        printf("   ERROR: Sequential result incorrect: %f\n", C[0]);
    }

    double avg_seq = 0;
    for (double t : times_seq) avg_seq += t;
    avg_seq /= ITERATIONS;

    double min_seq = *min_element(times_seq.begin(), times_seq.end());
    double max_seq = *max_element(times_seq.begin(), times_seq.end());
    double gflops_seq = (2.0 * N * N * N) / (avg_seq * 1e9);

    printf("   Average: %.3f ms (%.2f GFLOPS)\n", avg_seq * 1000, gflops_seq);
    printf("   Min:     %.3f ms (%.2f GFLOPS)\n", min_seq * 1000, (2.0 * N * N * N) / (min_seq * 1e9));
    printf("   Max:     %.3f ms\n\n", max_seq * 1000);

    // Benchmark Parallel
    omp_set_num_threads(16);
    cout << "2. Parallel OpenMP (16 threads)\n";

    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        matmul_parallel(A, B, C);
    }

    vector<double> times_par;
    for (int i = 0; i < ITERATIONS; i++) {
        auto start = high_resolution_clock::now();
        matmul_parallel(A, B, C);
        auto end = high_resolution_clock::now();

        duration<double> elapsed = end - start;
        times_par.push_back(elapsed.count());
    }

    // Verify
    if (C[0] != 1024.0f) {
        printf("   ERROR: Parallel result incorrect: %f\n", C[0]);
    }

    double avg_par = 0;
    for (double t : times_par) avg_par += t;
    avg_par /= ITERATIONS;

    double min_par = *min_element(times_par.begin(), times_par.end());
    double max_par = *max_element(times_par.begin(), times_par.end());
    double gflops_par = (2.0 * N * N * N) / (avg_par * 1e9);

    printf("   Average: %.3f ms (%.2f GFLOPS)\n", avg_par * 1000, gflops_par);
    printf("   Min:     %.3f ms (%.2f GFLOPS)\n", min_par * 1000, (2.0 * N * N * N) / (min_par * 1e9));
    printf("   Max:     %.3f ms\n\n", max_par * 1000);

    cout << "=========================================\n";
    printf("   Speedup: %.2fx (average)\n", avg_seq / avg_par);
    printf("   Best case speedup: %.2fx\n", min_seq / min_par);
    cout << "=========================================\n";

    free(A);
    free(B);
    free(C);

    return 0;
}
