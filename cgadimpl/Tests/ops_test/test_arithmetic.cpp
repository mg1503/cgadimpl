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

// Helper to compare tensors
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

void test_add_sub() {
    std::cout << "[Test] Add/Sub... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    Value a = make_tensor(Tensor::ones(Shape{{2}}, opts), "a");
    a.node->requires_grad_flag_ = true;
    a.node->grad = Tensor::zeros(Shape{{2}}, opts);
    
    Value b = make_tensor(Tensor::ones(Shape{{2}}, opts), "b");
    b.node->requires_grad_flag_ = true;
    b.node->grad = Tensor::zeros(Shape{{2}}, opts);
    
    Value c = add(a, b); // [2, 2]
    Value d = sub(c, a); // [1, 1]
    
    Tensor expected_d = Tensor::ones(Shape{{2}}, opts);
    if (!compare_tensors(d.node->value, expected_d)) {
        std::cout << "❌ Failed Forward\n";
        return;
    }
    
    backward(sum(d));
    // d = (a+b) - a = b.
    // grad w.r.t a: d(b)/da = 0? No, d = a+b-a.
    // c = a+b. d = c-a.
    // dL/dd = 1.
    // dL/da = dL/dd * dd/da = 1 * (dc/da - 1) = 1 * (1 - 1) = 0.
    // dL/db = dL/dd * dd/db = 1 * (dc/db) = 1 * 1 = 1.
    
    // Check grad a is 0
    Tensor expected_grad_a = Tensor::zeros(Shape{{2}}, opts);
    if (!compare_tensors(a.node->grad, expected_grad_a)) {
        std::cout << "❌ Failed Backward (a)\n";
        return;
    }
    // Check grad b is 1
    Tensor expected_grad_b = Tensor::ones(Shape{{2}}, opts);
    if (!compare_tensors(b.node->grad, expected_grad_b)) {
        std::cout << "❌ Failed Backward (b)\n";
        return;
    }
    std::cout << "✅ Passed\n";
}

void test_mul_div() {
    std::cout << "[Test] Mul/Div... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    std::vector<float> data_a = {2.0f, 4.0f};
    Value a = make_tensor(create_tensor(data_a.data(), Shape{{2}}, opts), "a");
    a.node->requires_grad_flag_ = true;
    a.node->grad = Tensor::zeros(Shape{{2}}, opts);
    
    Value b = make_tensor(create_tensor(data_a.data(), Shape{{2}}, opts), "b");
    b.node->requires_grad_flag_ = true;
    b.node->grad = Tensor::zeros(Shape{{2}}, opts);
    
    Value c = mul(a, b); // [4, 16]
    Value d = div(c, a); // [2, 4] == b
    
    if (!compare_tensors(d.node->value, b.node->value)) {
        std::cout << "❌ Failed Forward\n";
        return;
    }
    
    backward(sum(d));
    // d = (a*b)/a = b.
    // grad a should be 0 (symbolically), but numerically:
    // c = a*b. d = c/a.
    // dL/da = dL/dd * ( (dc/da * a - c * 1) / a^2 )
    //       = 1 * ( (b*a - a*b) / a^2 ) = 0.
    // dL/db = dL/dd * (dc/db / a) = 1 * (a/a) = 1.
    
    Tensor expected_grad_a = Tensor::zeros(Shape{{2}}, opts);
    if (!compare_tensors(a.node->grad, expected_grad_a)) {
        std::cout << "❌ Failed Backward (a)\n";
        return;
    }
    Tensor expected_grad_b = Tensor::ones(Shape{{2}}, opts);
    if (!compare_tensors(b.node->grad, expected_grad_b)) {
        std::cout << "❌ Failed Backward (b)\n";
        return;
    }
    std::cout << "✅ Passed\n";
}

void test_fmab() {
    std::cout << "[Test] FMAB (Fused Mul Add)... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    // y = a@b + c (matmul + add)
    Value a = make_tensor(Tensor::ones(Shape{{2, 2}}, opts), "a");
    a.node->requires_grad_flag_ = true;
    a.node->grad = Tensor::zeros(Shape{{2, 2}}, opts);
    
    Value b = make_tensor(Tensor::ones(Shape{{2, 2}}, opts), "b");
    b.node->requires_grad_flag_ = true;
    b.node->grad = Tensor::zeros(Shape{{2, 2}}, opts);
    
    Value c = make_tensor(Tensor::ones(Shape{{2, 2}}, opts), "c");
    c.node->requires_grad_flag_ = true;
    c.node->grad = Tensor::zeros(Shape{{2, 2}}, opts);
    
    Value y = fmab(a, b, c);
    
    // 1@1 + 1 = 2+1 = 3? No.
    // ones(2,2) @ ones(2,2) = [[2,2],[2,2]]. + ones(2,2) = [[3,3],[3,3]].
    
    Tensor expected = create_tensor(std::vector<float>{3.0f, 3.0f, 3.0f, 3.0f}.data(), Shape{{2, 2}}, opts);
    if (!compare_tensors(y.node->value, expected)) {
        std::cout << "❌ Failed Forward\n";
        return;
    }
    
    backward(sum(y));
    std::cout << "✅ Passed\n";
}

void test_edge_cases() {
    std::cout << "[Test] Edge Cases (NaN/Inf)... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    
    float nan = std::numeric_limits<float>::quiet_NaN();
    float inf = std::numeric_limits<float>::infinity();
    
    // Test NaN propagation
    Value a = make_tensor(create_tensor(std::vector<float>{nan, 1.0f}.data(), Shape{{2}}, opts), "a");
    Value b = make_tensor(create_tensor(std::vector<float>{1.0f, 1.0f}.data(), Shape{{2}}, opts), "b");
    Value c = add(a, b);
    
    Tensor c_val = c.node->value.to_cpu();
    if (!std::isnan(c_val.data<float>()[0])) {
        std::cout << "❌ Failed NaN propagation\n";
        return;
    }
    
    // Test Inf
    Value d = make_tensor(create_tensor(std::vector<float>{inf, 1.0f}.data(), Shape{{2}}, opts), "d");
    Value e = add(d, b);
    Tensor e_val = e.node->value.to_cpu();
    if (!std::isinf(e_val.data<float>()[0])) {
        std::cout << "❌ Failed Inf propagation\n";
        return;
    }
    
    std::cout << "✅ Passed\n";
}

int main() {
    std::cout << "Running Arithmetic Tests...\n";
    test_add_sub();
    test_mul_div();
    test_fmab();
    test_edge_cases();
    return 0;
}
