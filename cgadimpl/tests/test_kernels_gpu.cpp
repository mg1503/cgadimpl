// =========================================================
// FILE: cgadimpl/tests/test_kernels_gpu.cpp
// =========================================================
#include "ad/kernels_api.hpp"
#include "tensor.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Re-using the same helper from the CPU test
void check_tensors_close(const ag::Tensor& a, const ag::Tensor& b, const std::string& label, float epsilon = 1e-4f) {
    assert(a.shape() == b.shape());
    for (int r = 0; r < a.rows(); ++r) {
        for (int c = 0; c < a.cols(); ++c) {
            if (std::abs(a(r, c) - b(r, c)) > epsilon) {
                std::cerr << "FAIL: " << label << " mismatch at (" << r << "," << c << ")\n";
                std::cerr << "Tensor A (ref):\n" << a << "\n";
                std::cerr << "Tensor B (out):\n" << b << "\n";
                throw std::runtime_error("Tensor check failed for " + label);
            }
        }
    }
    std::cout << "PASS: " << label << "\n";
}

// --- GPU Memory Helpers ---
float* to_gpu(const ag::Tensor& t) {
    float* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, t.numel() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ptr, t.data(), t.numel() * sizeof(float), cudaMemcpyHostToDevice));
    return d_ptr;
}

ag::Tensor from_gpu(const float* d_ptr, int rows, int cols) {
    ag::Tensor t(rows, cols);
    CUDA_CHECK(cudaMemcpy(t.data(), d_ptr, t.numel() * sizeof(float), cudaMemcpyDeviceToHost));
    return t;
}

// --- Test Functions ---

void test_gpu_add() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(8, 8, 1);
    ag::Tensor b_cpu = ag::Tensor::randn(8, 8, 2);
    ag::Tensor ref = a_cpu + b_cpu;

    float *a_gpu = to_gpu(a_cpu), *b_gpu = to_gpu(b_cpu), *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));

    K.add(a_gpu, b_gpu, c_gpu, ref.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(c_gpu, 8, 8);
    check_tensors_close(ref, out, "test_gpu_add");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
}

void test_gpu_matmul() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(8, 16, 3);
    ag::Tensor b_cpu = ag::Tensor::randn(16, 8, 4);
    ag::Tensor ref = ag::Tensor::matmul(a_cpu, b_cpu);

    float *a_gpu = to_gpu(a_cpu), *b_gpu = to_gpu(b_cpu), *c_gpu;
    CUDA_CHECK(cudaMalloc(&c_gpu, ref.numel() * sizeof(float)));

    K.matmul(a_gpu, b_gpu, c_gpu, 8, 16, 8, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor out = from_gpu(c_gpu, 8, 8);
    check_tensors_close(ref, out, "test_gpu_matmul");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
}

void test_gpu_vjp_add() {
    auto& K = ag::kernels::cuda();
    ag::Tensor gy_cpu = ag::Tensor::randn(8, 8, 5);
    ag::Tensor ga_ref = gy_cpu; // vjp_add just passes gradient through
    ag::Tensor gb_ref = gy_cpu;

    ag::Tensor ga_cpu_init = ag::Tensor::zeros(8, 8);
    ag::Tensor gb_cpu_init = ag::Tensor::zeros(8, 8);

    float *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu = to_gpu(ga_cpu_init);
    float *gb_gpu = to_gpu(gb_cpu_init);

    K.vjp_add(ga_gpu, gb_gpu, gy_gpu, gy_cpu.numel(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 8, 8);
    ag::Tensor gb_out = from_gpu(gb_gpu, 8, 8);

    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_add (gA)");
    check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_add (gB)");

    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
    CUDA_CHECK(cudaFree(gb_gpu));
}

void test_gpu_vjp_matmul() {
    auto& K = ag::kernels::cuda();
    ag::Tensor a_cpu = ag::Tensor::randn(8, 16, 6);
    ag::Tensor b_cpu = ag::Tensor::randn(16, 8, 7);
    ag::Tensor gy_cpu = ag::Tensor::randn(8, 8, 8);

    // Reference calculation on CPU
    ag::Tensor ga_ref = ag::Tensor::matmul(gy_cpu, ag::Tensor::transpose(b_cpu));
    ag::Tensor gb_ref = ag::Tensor::matmul(ag::Tensor::transpose(a_cpu), gy_cpu);

    float *a_gpu = to_gpu(a_cpu), *b_gpu = to_gpu(b_cpu), *gy_gpu = to_gpu(gy_cpu);
    float *ga_gpu, *gb_gpu;
    CUDA_CHECK(cudaMalloc(&ga_gpu, ga_ref.numel() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gb_gpu, gb_ref.numel() * sizeof(float)));

    K.vjp_matmul(ga_gpu, gb_gpu, gy_gpu, a_gpu, b_gpu, 8, 16, 8, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    ag::Tensor ga_out = from_gpu(ga_gpu, 8, 16);
    ag::Tensor gb_out = from_gpu(gb_gpu, 16, 8);

    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_matmul (gA)");
    check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_matmul (gB)");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(gy_gpu));
    CUDA_CHECK(cudaFree(ga_gpu));
    CUDA_CHECK(cudaFree(gb_gpu));
}

int main() {
    std::cout << "=== Running GPU Kernel Tests ===\n";
    try {
        #if defined(_WIN32)
            const char* plugin_path = "./agkernels_cuda.dll";
        #elif defined(__APPLE__)
            const char* plugin_path = "./libagkernels_cuda.dylib";
        #else
            const char* plugin_path = "./libagkernels_cuda.so";
        #endif

        std::cout << "Loading GPU plugin from: " << plugin_path << "\n";
        ag::kernels::load_cuda_plugin(plugin_path);

        test_gpu_add();
        test_gpu_matmul();
        test_gpu_vjp_add();
        test_gpu_vjp_matmul();

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nAll GPU kernel tests passed successfully!\n";
    return 0;
}