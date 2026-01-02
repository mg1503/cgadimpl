#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#include "ad/autodiff/autodiff.hpp"
#include "ad/ops/ops.hpp"

using namespace ag;

Tensor create_tensor(const float* data, Shape shape, TensorOptions opts) {
    Tensor t = Tensor::zeros(shape, opts);
    t.set_data(data, t.numel());
    return t;
}

bool compare_tensors(const Tensor& a, const Tensor& b, float tol = 1e-4) {
    if (a.numel() != b.numel()) return false;
    Tensor a_cpu = a.to_cpu();
    Tensor b_cpu = b.to_cpu();
    const float* data_a = a_cpu.data<float>();
    const float* data_b = b_cpu.data<float>();
    for (size_t i = 0; i < a.numel(); ++i) {
        if (std::abs(data_a[i] - data_b[i]) > tol) return false;
    }
    return true;
}

void test_sin_cos() {
    std::cout << "[Test] Sin/Cos... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    Value x = make_tensor(Tensor::zeros(Shape{{1}}, opts), "x");
    x.node->requires_grad_flag_ = true;
    x.node->grad = Tensor::zeros(Shape{{1}}, opts);
    
    Value s = sin(x); // 0
    Value c = cos(x); // 1
    
    Tensor expected_s = create_tensor(std::vector<float>{0.0f}.data(), Shape{{1}}, opts);
    Tensor expected_c = create_tensor(std::vector<float>{1.0f}.data(), Shape{{1}}, opts);
    
    if (!compare_tensors(s.node->value, expected_s) || !compare_tensors(c.node->value, expected_c)) {
        std::cout << "❌ Failed Forward\n";
        return;
    }
    
    backward(s + c);
    // d(sin+cos)/dx = cos - sin. At 0: 1 - 0 = 1.
    
    Tensor expected_grad = create_tensor(std::vector<float>{1.0f}.data(), Shape{{1}}, opts);
    if (!compare_tensors(x.node->grad, expected_grad)) {
        std::cout << "❌ Failed Backward\n";
        return;
    }
    std::cout << "  Passed\n";
}

int main() {
    std::cout << "Running Trigonometry Tests...\n";
    test_sin_cos();
    return 0;
}
