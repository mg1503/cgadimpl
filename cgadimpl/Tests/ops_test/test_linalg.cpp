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

void test_matmul() {
    std::cout << "[Test] Matmul... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    // A: [2, 3], B: [3, 2]
    Value A = make_tensor(Tensor::ones(Shape{{2, 3}}, opts), "A");
    A.node->requires_grad_flag_ = true;
    A.node->grad = Tensor::zeros(Shape{{2, 3}}, opts);
    
    Value B = make_tensor(Tensor::ones(Shape{{3, 2}}, opts), "B");
    B.node->requires_grad_flag_ = true;
    B.node->grad = Tensor::zeros(Shape{{3, 2}}, opts);
    
    Value C = matmul(A, B); // [2, 2], each element = 3
    
    Tensor expected = create_tensor(std::vector<float>{3, 3, 3, 3}.data(), Shape{{2, 2}}, opts);
    if (!compare_tensors(C.node->value, expected)) {
        std::cout << "❌ Failed Forward\n";
        return;
    }
    
    backward(sum(C));
    // C = AB. L = sum(C).
    // dL/dA = dL/dC * B^T = 1 * B^T. B is all ones [3, 2]. B^T is [2, 3] all ones.
    // But wait, dL/dC is all ones [2, 2].
    // dL/dA = [2, 2] * [2, 3] = [2, 3] where each element is sum(row of ones) = 2.
    
    Tensor expected_grad_A = create_tensor(std::vector<float>{2, 2, 2, 2, 2, 2}.data(), Shape{{2, 3}}, opts);
    if (!compare_tensors(A.node->grad, expected_grad_A)) {
        std::cout << "❌ Failed Backward (A)\n";
        return;
    }
    std::cout << "  Passed\n";
}

void test_transpose() {
    std::cout << "[Test] Transpose... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    Value A = make_tensor(Tensor::ones(Shape{{2, 3}}, opts), "A");
    A.node->requires_grad_flag_ = true;
    A.node->grad = Tensor::zeros(Shape{{2, 3}}, opts);
    
    Value B = transpose(A); // [3, 2]
    
    if (B.node->value.shape().dims != std::vector<int64_t>{3, 2}) {
        std::cout << "❌ Failed Forward Shape\n";
        return;
    }
    
    backward(sum(B));
    // L = sum(A^T). dL/dA = ones(2, 3).
    Tensor expected_grad = Tensor::ones(Shape{{2, 3}}, opts);
    if (!compare_tensors(A.node->grad, expected_grad)) {
        std::cout << "❌ Failed Backward. Expected: Ones. Got: ";
        if (A.node->grad.numel() == 0) std::cout << "<empty>";
        else {
             Tensor g = A.node->grad.to_cpu();
             const float* d = g.data<float>();
             for(size_t i=0; i<g.numel(); ++i) std::cout << d[i] << " ";
        }
        std::cout << "\n";
        return;
    }
    std::cout << "  Passed\n";
}

void test_linear() {
    std::cout << "[Test] Linear (Fused)... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    // y = xW^T + b
    Value x = make_tensor(create_tensor(std::vector<float>{1.0f, 2.0f}.data(), Shape{{1, 2}}, opts), "x");
    x.node->requires_grad_flag_ = true;
    x.node->grad = Tensor::zeros(Shape{{1, 2}}, opts);
    
    Value W = make_tensor(create_tensor(std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f}.data(), Shape{{3, 2}}, opts), "W");
    W.node->requires_grad_flag_ = true;
    W.node->grad = Tensor::zeros(Shape{{3, 2}}, opts);
    
    Value b = make_tensor(create_tensor(std::vector<float>{1.0f, 1.0f, 1.0f}.data(), Shape{{3}}, opts), "b");
    b.node->requires_grad_flag_ = true;
    b.node->grad = Tensor::zeros(Shape{{3}}, opts);
    
    Value y = linear(x, W, b);
    
    Tensor expected = create_tensor(std::vector<float>{4.0f, 7.0f, 10.0f}.data(), Shape{{1, 3}}, opts);
    if (!compare_tensors(y.node->value, expected)) {
        std::cout << "❌ Failed Forward\n";
        return;
    }
    
    backward(sum(y));
    std::cout << "  Passed\n";
}

int main() {
    std::cout << "Running Linear Algebra Tests...\n";
    test_matmul();
    test_transpose();
    test_linear();
    return 0;
}
