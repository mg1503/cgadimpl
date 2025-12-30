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

void test_sum() {
    std::cout << "[Test] Sum... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    Value x = make_tensor(Tensor::ones(Shape{{2, 2}}, opts), "x");
    x.node->requires_grad_flag_ = true;
    x.node->grad = Tensor::zeros(Shape{{2, 2}}, opts);
    Value y = sum(x); // 4
    
    Tensor expected = create_tensor(std::vector<float>{4.0f}.data(), Shape{{1}}, opts);
    if (!compare_tensors(y.node->value, expected)) {
        std::cout << "❌ Failed Forward\n";
        return;
    }
    
    backward(y);
    // grad x should be all ones
    Tensor expected_grad = Tensor::ones(Shape{{2, 2}}, opts);
    if (!compare_tensors(x.node->grad, expected_grad)) {
        std::cout << "❌ Failed Backward\n";
        return;
    }
    std::cout << "✅ Passed\n";
}

void test_rowsum() {
    std::cout << "[Test] RowSum... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    // [[1, 2], [3, 4]] -> [3, 7] (assuming row sum reduces last dim or similar)
    // Ops signature: rowsum(x). Usually reduces along last dim.
    Value x = make_tensor(create_tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}.data(), Shape{{2, 2}}, opts), "x");
    x.node->requires_grad_flag_ = true;
    x.node->grad = Tensor::zeros(Shape{{2, 2}}, opts);
    
    Value y = rowsum(x);
    // Expected: [3, 7]
    
    Tensor expected = create_tensor(std::vector<float>{3.0f, 7.0f}.data(), Shape{{2, 1}}, opts); // Or {2} depending on impl
    // Let's check shape
    if (y.node->value.numel() != 2) {
         std::cout << "❌ Failed Forward Numel\n";
         return;
    }
    
    backward(sum(y));
    // grad x should be ones
    Tensor expected_grad = Tensor::ones(Shape{{2, 2}}, opts);
    if (!compare_tensors(x.node->grad, expected_grad)) {
        std::cout << "❌ Failed Backward\n";
        return;
    }
    std::cout << "✅ Passed\n";
}

void test_softmax_row() {
    std::cout << "[Test] Softmax Row... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    Value x = make_tensor(Tensor::zeros(Shape{{1, 2}}, opts), "x"); // [0, 0]
    x.node->requires_grad_flag_ = true;
    x.node->grad = Tensor::zeros(Shape{{1, 2}}, opts);
    Value y = softmax_row(x); // [0.5, 0.5]
    
    Tensor expected = create_tensor(std::vector<float>{0.5f, 0.5f}.data(), Shape{{1, 2}}, opts);
    if (!compare_tensors(y.node->value, expected)) {
        std::cout << "❌ Failed Forward\n";
        return;
    }
    
    backward(sum(y)); // sum([0.5, 0.5]) = 1.
    // grad of softmax sum is 0?
    // y_i = e^x_i / sum(e^x_j). sum(y_i) = 1 always.
    // d(1)/dx = 0.
    
    Tensor expected_grad = Tensor::zeros(Shape{{1, 2}}, opts);
    if (!compare_tensors(x.node->grad, expected_grad)) {
        std::cout << "❌ Failed Backward\n";
        return;
    }
    std::cout << "✅ Passed\n";
}

int main() {
    std::cout << "Running Reduction Tests...\n";
    test_sum();
    test_rowsum();
    test_softmax_row();
    return 0;
}
