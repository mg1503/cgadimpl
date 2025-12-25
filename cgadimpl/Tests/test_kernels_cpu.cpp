#include "ad/ag_all.hpp" // Includes TensorLib.h and brings in namespaces
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <string>

// Use the correct namespaces as defined in your project
using namespace OwnTensor;
using namespace ag;

// Helper now uses types from the correct namespaces
void check_tensors_close(const Tensor& a, const Tensor& b, const std::string& label, float epsilon = 1e-5f) {
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
            debug::print_tensor("Tensor A", a);
            debug::print_tensor("Tensor B", b);
            throw std::runtime_error("Tensor check failed for " + label);
        }
    }
    std::cout << "PASS: " << label << "\n";
}

void test_cpu_relu() {
    auto& K = kernels::cpu();
    assert(K.relu != nullptr);

    // Use the modern API to create a CPU tensor from the correct namespace
    auto opts = TensorOptions().with_device(Device::CPU);
    Tensor x = Tensor::randn(Shape{{4, 4}}, opts);
    
    float* x_data = x.data<float>();
    x_data[0] = -5.0f;
    x_data[5] = 0.0f;
    x_data[10] = -0.1f;

    // Use the framework's high-level `relu` op for the reference calculation
    Value x_val = make_tensor(x);
    Tensor y_ref = relu(x_val).val();

    Tensor y_out(x.shape(), options(x));
    K.relu(x.data<float>(), y_out.data<float>(), x.numel());

    check_tensors_close(y_ref, y_out, "test_cpu_relu");
}

void test_cpu_matmul() {
    auto& K = kernels::cpu();
    assert(K.matmul != nullptr);
    
    auto opts = TensorOptions().with_device(Device::CPU);
    Tensor a = Tensor::randn(Shape{{8, 16}}, opts);
    Tensor b = Tensor::randn(Shape{{16, 8}}, opts);

    // Use the underlying OwnTensor::matmul for the reference calculation
    Tensor c_ref = OwnTensor::matmul(a, b);
    Tensor c_out(Shape{{8, 8}}, options(a));

    K.matmul(a.data<float>(), b.data<float>(), c_out.data<float>(), 8, 16, 8);

    check_tensors_close(c_ref, c_out, "test_cpu_matmul");
}

int main() {
    std::cout << "=== Running CPU Kernel Tests ===\n";
    try {
        #if defined(_WIN32)
            const char* plugin_path = "./agkernels_cpu.dll";
        #elif defined(__APPLE__)
            const char* plugin_path = "./libagkernels_cpu.dylib";
        #else
            const char* plugin_path = "./libagkernels_cpu.so";
        #endif

        std::cout << "Loading CPU plugin from: " << plugin_path << "\n";
        kernels::load_cpu_plugin(plugin_path);

        test_cpu_relu();
        test_cpu_matmul();

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nAll CPU kernel tests passed successfully!\n";
    return 0;
}