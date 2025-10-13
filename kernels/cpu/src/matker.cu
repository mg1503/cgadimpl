#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>


#define TILE 4

__global__ void tile_matrix_multiply(float* A, float* B, float* C, int width)
{
    __shared__ float shareA[TILE][TILE];
    __shared__ float shareB[TILE][TILE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float temp = 0.0f;

    for (int i = 0; i < width / TILE; ++i) {
        shareA[ty][tx] = A[row * width + (i * TILE + tx)];
        shareB[ty][tx] = B[(i * TILE + ty) * width + col];
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            temp += shareA[ty][k] * shareB[k][tx];

        __syncthreads();
    }

    if (row < width && col < width)
        C[row * width + col] = temp;
}


void run_cuda_matrix(const float* A, const float* B, float* C, int width)
{
    float *d_A, *d_B, *d_C;
    int size = width * width * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE, TILE);
    dim3 numBlocks(width / TILE, width / TILE);

    tile_matrix_multiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, width);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}



__global__ void tile_gemm(float* A, float* B, float* C, int width)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float acc = 0.0f;

    for (int t = 0; t < width / TILE; ++t) {
        int a_idx = row * width + (t * TILE + tx);
        int b_idx = (t * TILE + ty) * width + col;

        As[ty][tx] = A[a_idx];
        Bs[ty][tx] = B[b_idx];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc = fmaf(As[ty][k], Bs[k][tx], acc);

        __syncthreads();
    }

    // Accumulate into existing C value instead of overwriting
    C[row * width + col] = fmaf(1.0f, acc, C[row * width + col]);
}


void run_cuda_gemm(const float* A, const float* B, float* C, int width)
{
    float *d_A, *d_B, *d_C;
    int size = width * width * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE, TILE);
    dim3 numBlocks(width / TILE, width / TILE);

    tile_gemm<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, width);
    cudaDeviceSynchronize();

        cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);



    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}




// int main()
// {
//     const int width = 4;
//     const int size = width * width * sizeof(float);

//     float h_A[width * width] = {
//         1, 2, 3, 4,
//         5, 6, 7, 8,
//         9, 10, 11, 12,
//         13, 14, 15, 16
//     };

//     float h_B[width * width] = {
//         1, 0, 0, 0,
//         0, 1, 0, 0,
//         0, 0, 1, 0,
//         0, 0, 0, 1
//     };

//     float h_C[width * width] = {0};

//     float *d_A, *d_B, *d_C;
//     cudaMalloc(&d_A, size);
//     cudaMalloc(&d_B, size);
//     cudaMalloc(&d_C, size);

//     cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

//     dim3 threadsPerBlock(TILE, TILE);           // 4×4 threads per block
//     dim3 numBlocks(width / TILE, width / TILE); // 1×1 for 4×4 input

//     tile_matrix_multiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, width);
//     cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

//     cudaDeviceSynchronize();


//     std::cout << "Result matrix C:\n";
//     for (int i = 0; i < width; ++i) {
//         for (int j = 0; j < width; ++j)
//             std::cout << h_C[i * width + j] << " ";
//         std::cout << "\n";
//     }

//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);

//     return 0;
// }