#pragma once
void run_cuda_matrix(const float* A, const float* B, float* C, int N);
void run_cuda_gemm(const float* A, const float* B, float* C, int N);