    // ====================================================================
    // FILE: kernels/gpu/mm_cublas.cu (The Final, Textbook-Correct Fix)
    // ====================================================================
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
    #include "ad/kernels_api.hpp"

    // --- Forward Pass: C = A @ B ---
    // A is (M, K), B is (K, N), C is (M, N). All are row-major.
    void mm_cuda(const float* A, const float* B, float* C,
                int M, int K, int N, ag_cuda_stream_t s) {
        static thread_local cublasHandle_t handle = nullptr;
        if (!handle) cublasCreate(&handle);
        cublasSetStream(handle, (cudaStream_t)s);

        const float alpha = 1.f, beta = 0.f;

        // Row-major C(M,N) = A(M,K) @ B(K,N) is equivalent to
        // column-major C_t(N,M) = B_t(N,K) @ A_t(K,M).
        // We pass B as the first matrix and A as the second, with NO transpose flags.
        // API: cublasSgemm(handle, transa, transb, m, n, k, alpha, A_ptr, lda, B_ptr, ldb, beta, C_ptr, ldc)
        cublasSgemm(handle,
            CUBLAS_OP_N,        // transa on B
            CUBLAS_OP_N,        // transb on A
            N, M, K,            // m=N, n=M, k=K
            &alpha,
            B, N,               // B(K,N) -> lda is cols = N
            A, K,               // A(M,K) -> ldb is cols = K
            &beta,
            C, N);              // C(M,N) -> ldc is cols = N
    }

    // --- Backward Pass (VJP) ---
    void vjp_matmul_cuda(float* gA, float* gB, const float* gy,
                        const float* A, const float* B,
                        int M, int K, int N, ag_cuda_stream_t s)
    {
        static thread_local cublasHandle_t handle = nullptr;
        if (!handle) cublasCreate(&handle);
        cublasSetStream(handle, (cudaStream_t)s);
        const float alpha = 1.f, beta = 0.f;

        // gA(M,K) = gy(M,N) @ B^T(N,K)
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, &alpha, B, N, gy, N, &beta, gA, K);

        // gB(K,N) = A^T(K,M) @ gy(M,N)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, &alpha, gy, N, A, K, &beta, gB, N);
    }