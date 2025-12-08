// OpenMP parallel matmul wrapper
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <chrono>

// 512x512 matrix multiplication with OpenMP + tiling + vectorization
void matmul_parallel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    const int N = 512;
    const int TILE = 64;  // Tile size for cache blocking

    // Initialize C to zero
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0f;
        }
    }

    // Tiled matrix multiplication with OpenMP parallelization
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i0 = 0; i0 < N; i0 += TILE) {
        for (int j0 = 0; j0 < N; j0 += TILE) {
            for (int k0 = 0; k0 < N; k0 += TILE) {
                // Tile boundaries
                int i_end = (i0 + TILE < N) ? i0 + TILE : N;
                int j_end = (j0 + TILE < N) ? j0 + TILE : N;
                int k_end = (k0 + TILE < N) ? k0 + TILE : N;

                for (int i = i0; i < i_end; i++) {
                    for (int k = k0; k < k_end; k++) {
                        float a_ik = A[i * N + k];
                        // This loop should auto-vectorize
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

    // Allocate aligned memory
    float *A = (float*)aligned_alloc(64, SIZE * sizeof(float));
    float *B = (float*)aligned_alloc(64, SIZE * sizeof(float));
    float *C = (float*)aligned_alloc(64, SIZE * sizeof(float));

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize matrices
    for (int i = 0; i < SIZE; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Run matmul
    matmul_parallel(A, B, C);

    // Verify result (first element should be 1024.0)
    float expected = 1024.0f;
    int ret = 0;

    if (C[0] != expected) {
        fprintf(stderr, "Result incorrect: got %f, expected %f\n", C[0], expected);
        ret = 1;
    }

    // Cleanup
    free(A);
    free(B);
    free(C);

    return ret;
}
