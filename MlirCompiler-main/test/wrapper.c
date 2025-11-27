#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

// Define the matrix dimensions from your MLIR code
#define N 4096
#define M 4096
#define K 4096

// --- C FUNCTION SIGNATURE FOR THE MLIR-GENERATED CODE ---
// The MLIR func @matmul_with_bias(%A, %B, %bias) -> %C_result is lowered to this:
// Each tensor<NxMxf32> becomes a 7-argument MemRef descriptor.
extern void matmul_with_bias(
    // A (Input 1: N x K)
    float *A_ptr, float *A_aligned, int64_t A_offset, int64_t A_size0, int64_t A_size1, int64_t A_stride0, int64_t A_stride1,
    // B (Input 2: K x M)
    float *B_ptr, float *B_aligned, int64_t B_offset, int64_t B_size0, int64_t B_size1, int64_t B_stride0, int64_t B_stride1,
    // Bias (Input 3: N x M)
    float *Bias_ptr, float *Bias_aligned, int64_t Bias_offset, int64_t Bias_size0, int64_t Bias_size1, int64_t Bias_stride0, int64_t Bias_stride1,
    // Result C (Output 4: N x M) - Passed as the output argument
    float *Result_ptr, float *Result_aligned, int64_t Result_offset, int64_t Result_size0, int64_t Result_size1, int64_t Result_stride0, int64_t Result_stride1
);


// Helper function to initialize buffers with test data
void init_buffer(float *buffer, int64_t size, float value) {
    for (int i = 0; i < size; ++i) {
        buffer[i] = value;
    }
}

// Helper function to print a small section of the result
void print_result_snippet(float *buffer, int rows, int cols) {
    printf("Result Snippet (Top-Left 4x4):\n");
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%8.2f", buffer[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    // Total number of elements in each 4096x4096 matrix
    const size_t total_elements = (size_t)N * M;
    const size_t buffer_bytes = total_elements * sizeof(float);

    // 1. Memory Allocation (aligned_alloc is preferred for AVX/OpenMP)
    float *A_data = (float*)aligned_alloc(64, buffer_bytes);
    float *B_data = (float*)aligned_alloc(64, buffer_bytes);
    float *Bias_data = (float*)aligned_alloc(64, buffer_bytes);
    float *C_data = (float*)aligned_alloc(64, buffer_bytes);

    if (!A_data || !B_data || !Bias_data || !C_data) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return 1;
    }

    // 2. Data Initialization (A=1.0, B=1.0, Bias=0.5)
    init_buffer(A_data, total_elements, 1.0f);
    init_buffer(B_data, total_elements, 1.0f);
    init_buffer(Bias_data, total_elements, 0.5f);
    init_buffer(C_data, total_elements, 0.0f);
    
    printf("--- Running MLIR Optimized MatMul (N=%d) ---\n", N);
    
    // 3. Time the execution
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // 4. CALL THE OPTIMIZED FUNCTION (28 arguments for 4 descriptors)
    matmul_with_bias(
        // A (Input 1: N x K)
        A_data, A_data, 0, N, K, K, 1,
        // B (Input 2: K x M)
        B_data, B_data, 0, K, M, M, 1,
        // Bias (Input 3: N x M)
        Bias_data, Bias_data, 0, N, M, M, 1,
        // Result C (Output 4: N x M)
        C_data, C_data, 0, N, M, M, 1
    );

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_time_ms = (end.tv_sec - start.tv_sec) * 1000.0;
    elapsed_time_ms += (end.tv_nsec - start.tv_nsec) / 1000000.0;

    // 5. Verification and Cleanup
    printf("\nExecution Time: %.3f ms\n", elapsed_time_ms);
    print_result_snippet(C_data, N, M);
    
    // Simple verification check (Expected: 4096 * 1.0 + 0.5 = 4096.5)
    float expected = 4096.5f;
    if (fabs(C_data[0] - expected) < 0.01) {
        printf("Verification: SUCCESS (C[0] is %.2f)\n", C_data[0]);
    } else {
        printf("Verification: FAILED (C[0] is %.2f, expected %.2f)\n", C_data[0], expected);
    }
    
    // Cleanup
    free(A_data);
    free(B_data);
    free(Bias_data);
    free(C_data);

    return 0;
}