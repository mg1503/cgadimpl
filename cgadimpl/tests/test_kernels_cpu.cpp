// =========================================================
// FILE: cgadimpl/tests/test_kernels_cpu.cpp
// =========================================================
#include "ad/kernels_api.hpp"
#include "tensor.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <string>

// A simple helper to check if two tensors are close enough
void check_tensors_close(const ag::Tensor& a, const ag::Tensor& b, const std::string& label, float epsilon = 1e-5f) {
    assert(a.shape() == b.shape());
    for (int r = 0; r < a.rows(); ++r) {
        for (int c = 0; c < a.cols(); ++c) {
            if (std::abs(a(r, c) - b(r, c)) > epsilon) {
                std::cerr << "FAIL: " << label << " mismatch at (" << r << "," << c << ")\n";
                std::cerr << "Tensor A:\n" << a << "\n";
                std::cerr << "Tensor B:\n" << b << "\n";
                throw std::runtime_error("Tensor check failed for " + label);
            }
        }
    }
    std::cout << "PASS: " << label << "\n";
}

void test_cpu_relu() {
    auto& K = ag::kernels::cpu();
    assert(K.relu != nullptr);

    ag::Tensor x = ag::Tensor::randn(4, 4, 123);
    x(0, 0) = -5.0f; x(1, 1) = 0.0f; x(2, 2) = -0.1f;

    ag::Tensor y_ref = ag::Tensor::relu(x);
    ag::Tensor y_out(4, 4);

    K.relu(x.data(), y_out.data(), x.numel());

    check_tensors_close(y_ref, y_out, "test_cpu_relu");
}

void test_cpu_matmul() {
    auto& K = ag::kernels::cpu();
    assert(K.matmul != nullptr);

    ag::Tensor a = ag::Tensor::randn(8, 16, 456);
    ag::Tensor b = ag::Tensor::randn(16, 8, 789);

    ag::Tensor c_ref = ag::Tensor::matmul(a, b);
    ag::Tensor c_out(8, 8);

    K.matmul(a.data(), b.data(), c_out.data(), 8, 16, 8);

    check_tensors_close(c_ref, c_out, "test_cpu_matmul");
}

int main() {
    std::cout << "=== Running CPU Kernel Tests ===\n";
    try {
        // Path depends on OS and where run.sh stages it.
        // Assuming we run from the core build directory.
        #if defined(_WIN32)
            const char* plugin_path = "./agkernels_cpu.dll";
        #elif defined(__APPLE__)
            const char* plugin_path = "./libagkernels_cpu.dylib";
        #else
            const char* plugin_path = "./libagkernels_cpu.so";
        #endif

        std::cout << "Loading CPU plugin from: " << plugin_path << "\n";
        ag::kernels::load_cpu_plugin(plugin_path);

        test_cpu_relu();
        test_cpu_matmul();

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nAll CPU kernel tests passed successfully!\n";
    return 0;
}