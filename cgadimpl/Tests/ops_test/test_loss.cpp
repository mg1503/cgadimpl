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

void test_mse() {
    std::cout << "[Test] MSE Loss... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    Value pred = make_tensor(create_tensor(std::vector<float>{1.0f, 2.0f}.data(), Shape{{2}}, opts), "pred");
    pred.node->requires_grad_flag_ = true;
    pred.node->grad = Tensor::zeros(Shape{{2}}, opts);
    
    Value target = make_tensor(create_tensor(std::vector<float>{1.0f, 1.0f}.data(), Shape{{2}}, opts), "target");
    target.node->requires_grad_flag_ = true;
    target.node->grad = Tensor::zeros(Shape{{2}}, opts);
    
    Value loss = mse_loss(pred, target);
    // MSE = mean((1-1)^2 + (2-1)^2) = mean(0 + 1) = 0.5
    
    Tensor expected = create_tensor(std::vector<float>{0.5f}.data(), Shape{{1}}, opts);
    if (!compare_tensors(loss.node->value, expected)) {
        std::cout << "❌ Failed Forward. Got " << loss.node->value.to_cpu().data<float>()[0] << "\n";
        return;
    }
    
    backward(loss);
    // dL/dpred = 2/N * (pred - target)
    // dL/dp1 = 2/2 * (1-1) = 0
    // dL/dp2 = 2/2 * (2-1) = 1
    
    Tensor expected_grad = create_tensor(std::vector<float>{0.0f, 1.0f}.data(), Shape{{2}}, opts);
    if (!compare_tensors(pred.node->grad, expected_grad)) {
        std::cout << "❌ Failed Backward\n";
        return;
    }
    std::cout << "  Passed\n";
}

void test_cross_entropy() {
    std::cout << "[Test] Cross Entropy... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    // Logits: [2.0, 1.0]
    // Softmax: exp(2)/(e^2+e^1) = 7.389 / (7.389+2.718) = 0.731
    //          exp(1)/(...) = 0.269
    // Target: [1, 0]
    // Loss = -log(0.731) = 0.313
    
    Value logits = make_tensor(create_tensor(std::vector<float>{2.0f, 1.0f}.data(), Shape{{1, 2}}, opts), "logits");
    logits.node->requires_grad_flag_ = true;
    logits.node->grad = Tensor::zeros(Shape{{1, 2}}, opts);
    
    Value target = make_tensor(create_tensor(std::vector<float>{1.0f, 0.0f}.data(), Shape{{1, 2}}, opts), "target");
    target.node->requires_grad_flag_ = true;
    target.node->grad = Tensor::zeros(Shape{{1, 2}}, opts);
    
    Value loss = cross_entropy_with_logits(logits, target);
    
    // Just check backward runs
    backward(loss);
    // Grad w.r.t logits = softmax(logits) - target
    // = [0.731 - 1, 0.269 - 0] = [-0.269, 0.269]
    
    Tensor grad = logits.node->grad.to_cpu();
    float g1 = grad.data<float>()[0];
    float g2 = grad.data<float>()[1];
    
    if (std::abs(g1 + 0.2689f) < 1e-3 && std::abs(g2 - 0.2689f) < 1e-3) {
        std::cout << "  Passed\n";
    } else {
        std::cout << "❌ Failed Backward. Got " << g1 << ", " << g2 << "\n";
    }
}

int main() {
    std::cout << "Running Loss Tests...\n";
    test_mse();
    test_cross_entropy();
    return 0;
}
