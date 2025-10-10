#pragma once

// This header provides the declarations for all kernel implementations
// that we want to benchmark.

extern "C" {
    void matmul_impl_naive(const float* A, const float* B, float* C, int M, int K, int N);
    void matmul_impl_optimized(const float* A, const float* B, float* C, int M, int K, int N);
}