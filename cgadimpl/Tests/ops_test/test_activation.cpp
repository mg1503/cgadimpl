#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

#include "ad/autodiff/autodiff.hpp"
#include "ad/ops/ops.hpp"
#include "ad/utils/debug.hpp"

using namespace ag;

Tensor create_tensor(const float* data, Shape shape, TensorOptions opts) {
    Tensor t = Tensor::zeros(shape, opts);
    t.set_data(data, t.numel());
    return t;
}

// Helper to compare tensors
bool compare_tensors(const Tensor& a, const Tensor& b, float tol = 1e-4) {
    if (a.numel() != b.numel()) return false;
    
    Tensor a_cpu = a.to_cpu();
    Tensor b_cpu = b.to_cpu();
    
    const float* data_a = a_cpu.data<float>();
    const float* data_b = b_cpu.data<float>();
    
    for (size_t i = 0; i < a.numel(); ++i) {
        if (std::abs(data_a[i] - data_b[i]) > tol) {
            std::cerr << "Mismatch at index " << i << ": " << data_a[i] << " vs " << data_b[i] << "\n";
            return false;
        }
    }
    return true;
}

void test_relu() {
    std::cout << "[Test] ReLU... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    // x = [-1, 0, 1, 2]
    std::vector<float> data = {-1.0f, 0.0f, 1.0f, 2.0f};
    Value x = make_tensor(create_tensor(data.data(), Shape{{4}}, opts), "x");
    x.node->requires_grad_flag_ = true; // Ensure gradients are computed
    x.node->grad = Tensor::zeros(Shape{{4}}, opts);
    
    Value y = relu(x);
    
    // Expected: [0, 0, 1, 2]
    std::vector<float> expected_data = {0.0f, 0.0f, 1.0f, 2.0f};
    Tensor expected = create_tensor(expected_data.data(), Shape{{4}}, opts);
    
    if (!compare_tensors(y.node->value, expected)) {
        std::cout << "❌ Failed Forward\n";
        return;
    }
    
    backward(sum(y));
    // Grad: [0, 0, 1, 1] (approx for 0)
    std::vector<float> expected_grad_data = {0.0f, 0.0f, 1.0f, 1.0f}; // VJP impl uses 0 for x=0
    
    Tensor expected_grad = create_tensor(expected_grad_data.data(), Shape{{4}}, opts);
    if (!compare_tensors(x.node->grad, expected_grad)) {
         std::cout << "❌ Failed Backward. Expected: ";
         for(auto v : expected_grad_data) std::cout << v << " ";
         std::cout << "\nGot: ";
         if (x.node->grad.numel() == 0) {
             std::cout << "<empty>";
         } else {
             Tensor g = x.node->grad.to_cpu();
             const float* d = g.data<float>();
             for(size_t i=0; i<g.numel(); ++i) std::cout << d[i] << " ";
         }
         std::cout << "\n";
         return;
    }

    std::cout << "✅ Passed\n";
}

void test_sigmoid() {
    std::cout << "[Test] Sigmoid... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    std::vector<float> data = {0.0f};
    Value x = make_tensor(create_tensor(data.data(), Shape{{1}}, opts), "x");
    x.node->requires_grad_flag_ = true;
    x.node->grad = Tensor::zeros(Shape{{1}}, opts);
    
    Value y = sigmoid(x);
    // sigmoid(0) = 0.5
    
    Tensor expected = create_tensor(std::vector<float>{0.5f}.data(), Shape{{1}}, opts);
    if (!compare_tensors(y.node->value, expected)) {
        std::cout << "❌ Failed Forward\n";
        return;
    }
    
    backward(y); // grad of sigmoid at 0 is sigmoid(0)*(1-sigmoid(0)) = 0.5*0.5 = 0.25
    Tensor expected_grad = create_tensor(std::vector<float>{0.25f}.data(), Shape{{1}}, opts);
    
    if (!compare_tensors(x.node->grad, expected_grad)) {
        std::cout << "❌ Failed Backward. Expected: 0.25\nGot: ";
        if (x.node->grad.numel() == 0) {
             std::cout << "<empty>";
        } else {
            Tensor g = x.node->grad.to_cpu();
            const float* d = g.data<float>();
            for(size_t i=0; i<g.numel(); ++i) std::cout << d[i] << " ";
        }
        std::cout << "\n";
        return;
    }
    std::cout << "✅ Passed\n";
}

void test_tanh() {
    std::cout << "[Test] Tanh... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    std::vector<float> data = {0.0f};
    Value x = make_tensor(create_tensor(data.data(), Shape{{1}}, opts), "x");
    x.node->requires_grad_flag_ = true;
    x.node->grad = Tensor::zeros(Shape{{1}}, opts);
    
    Value y = tanh(x); // tanh(0) = 0
    
    Tensor expected = create_tensor(std::vector<float>{0.0f}.data(), Shape{{1}}, opts);
    if (!compare_tensors(y.node->value, expected)) {
        std::cout << "❌ Failed Forward\n";
        return;
    }
    
    backward(y); // grad of tanh at 0 is 1 - tanh^2(0) = 1
    Tensor expected_grad = create_tensor(std::vector<float>{1.0f}.data(), Shape{{1}}, opts);
    
    if (!compare_tensors(x.node->grad, expected_grad)) {
        std::cout << "❌ Failed Backward. Expected: 1.0\nGot: ";
        if (x.node->grad.numel() == 0) {
             std::cout << "<empty>";
        } else {
            Tensor g = x.node->grad.to_cpu();
            const float* d = g.data<float>();
            for(size_t i=0; i<g.numel(); ++i) std::cout << d[i] << " ";
        }
        std::cout << "\n";
        return;
    }
    std::cout << "✅ Passed\n";
}

void test_edge_cases() {
    std::cout << "[Test] Edge Cases (NaN/Inf)... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    
    float nan = std::numeric_limits<float>::quiet_NaN();
    
    // Test ReLU with NaN
    Value x = make_tensor(create_tensor(std::vector<float>{nan}.data(), Shape{{1}}, opts), "x");
    Value y = relu(x);
    
    Tensor y_val = y.node->value.to_cpu();
    if (!std::isnan(y_val.data<float>()[0])) {
        std::cout << "❌ Failed NaN propagation in ReLU\n";
        return;
    }
    
    std::cout << "✅ Passed\n";
}

int main() {
    std::cout << "Running Activation Tests...\n";
    test_relu();
    test_sigmoid();
    test_tanh();
    test_edge_cases();
    return 0;
}
