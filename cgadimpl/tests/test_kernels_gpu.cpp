#include "ad/ag_all.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

// Use the correct namespaces as defined in your project
using namespace OwnTensor;
using namespace ag;

// Helper updated to use types from the correct namespaces
void check_tensors_close(const Tensor& a, const Tensor& b, const std::string& label, float epsilon = 1e-4f) {
    if (a.shape().dims != b.shape().dims) {
        throw std::runtime_error(label + ": Shape mismatch.");
    }
    Tensor a_cpu = a.to_cpu();
    Tensor b_cpu = b.to_cpu();
    const float* a_data = a_cpu.data<float>();
    const float* b_data = b_cpu.data<float>();

    for (size_t i = 0; i < a.numel(); ++i) {
        if (std::abs(a_data[i] - b_data[i]) > epsilon) {
            std::cerr << "FAIL: " << label << " mismatch at index " << i << "\n";
            debug::print_tensor("Tensor A (ref)", a);
            debug::print_tensor("Tensor B (out)", b);
            throw std::runtime_error("Tensor check failed for " + label);
        }
    }
    std::cout << "PASS: " << label << "\n";
}

// --- Test Functions ---

void test_gpu_add() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(DeviceIndex(Device::CUDA));

    Tensor a_cpu = Tensor::randn(Shape{{8, 8}}, cpu_opts);
    Tensor b_cpu = Tensor::randn(Shape{{8, 8}}, cpu_opts);
    Tensor ref = a_cpu + b_cpu;

    Tensor a_gpu = a_cpu.to(gpu_opts.device);
    Tensor b_gpu = b_cpu.to(gpu_opts.device);
    Tensor c_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));

    K.add(a_gpu.data<float>(), b_gpu.data<float>(), c_gpu.data<float>(), ref.numel(), nullptr);
    cudaDeviceSynchronize();

    Tensor out_cpu = c_gpu.to_cpu();
    check_tensors_close(ref, out_cpu, "test_gpu_add");
}

void test_gpu_matmul() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(DeviceIndex(Device::CUDA));

    Tensor a_cpu = Tensor::randn(Shape{{8, 16}}, cpu_opts);
    Tensor b_cpu = Tensor::randn(Shape{{16, 8}}, cpu_opts);
    Tensor ref = OwnTensor::matmul(a_cpu, b_cpu);

    Tensor a_gpu = a_cpu.to(gpu_opts.device);
    Tensor b_gpu = b_cpu.to(gpu_opts.device);
    Tensor c_gpu(ref.shape(), options(ref).with_device(gpu_opts.device));
    
    K.matmul(a_gpu.data<float>(), b_gpu.data<float>(), c_gpu.data<float>(), 8, 16, 8, nullptr);
    cudaDeviceSynchronize();

    Tensor out_cpu = c_gpu.to_cpu();
    check_tensors_close(ref, out_cpu, "test_gpu_matmul");
}

void test_gpu_vjp_add() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(DeviceIndex(Device::CUDA));

    Tensor gy_cpu = Tensor::randn(Shape{{8, 8}}, cpu_opts);
    Tensor ga_ref = gy_cpu; // vjp_add just passes gradient through
    Tensor gb_ref = gy_cpu;

    Tensor ga_init = Tensor::zeros(Shape{{8, 8}}, gpu_opts);
    Tensor gb_init = Tensor::zeros(Shape{{8, 8}}, gpu_opts);
    Tensor gy_gpu = gy_cpu.to(gpu_opts.device);

    K.vjp_add(ga_init.data<float>(), gb_init.data<float>(), gy_gpu.data<float>(), gy_cpu.numel(), nullptr);
    cudaDeviceSynchronize();

    Tensor ga_out = ga_init.to_cpu();
    Tensor gb_out = gb_init.to_cpu();

    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_add (gA)");
    check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_add (gB)");
}

void test_gpu_vjp_matmul() {
    auto& K = kernels::cuda();
    auto cpu_opts = TensorOptions().with_device(Device::CPU);
    auto gpu_opts = TensorOptions().with_device(DeviceIndex(Device::CUDA));

    Tensor a_cpu = Tensor::randn(Shape{{8, 16}}, cpu_opts);
    Tensor b_cpu = Tensor::randn(Shape{{16, 8}}, cpu_opts);
    Tensor gy_cpu = Tensor::randn(Shape{{8, 8}}, cpu_opts);

    // Reference calculation on CPU
    Tensor ga_ref = OwnTensor::matmul(gy_cpu, b_cpu.t());
    Tensor gb_ref = OwnTensor::matmul(a_cpu.t(), gy_cpu);

    Tensor a_gpu = a_cpu.to(gpu_opts.device);
    Tensor b_gpu = b_cpu.to(gpu_opts.device);
    Tensor gy_gpu = gy_cpu.to(gpu_opts.device);
    
    Tensor ga_gpu(ga_ref.shape(), options(ga_ref).with_device(gpu_opts.device));
    Tensor gb_gpu(gb_ref.shape(), options(gb_ref).with_device(gpu_opts.device));
    
    K.vjp_matmul(ga_gpu.data<float>(), gb_gpu.data<float>(), gy_gpu.data<float>(), 
                 a_gpu.data<float>(), b_gpu.data<float>(), 8, 16, 8, nullptr);
    cudaDeviceSynchronize();

    Tensor ga_out = ga_gpu.to_cpu();
    Tensor gb_out = gb_gpu.to_cpu();

    check_tensors_close(ga_ref, ga_out, "test_gpu_vjp_matmul (gA)");
    check_tensors_close(gb_ref, gb_out, "test_gpu_vjp_matmul (gB)");
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
        kernels::load_cuda_plugin(plugin_path);

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