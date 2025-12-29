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

void test_attention_op() {
    std::cout << "[Test] Attention Op... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    // Simple dummy test since attention logic is complex
    // Assuming attention(q, k, v, mask) or similar signature from ops.cpp: attention(a, b, c, d)
    
    Value q = make_tensor(Tensor::randn(Shape{{1, 4, 8}}, opts), "q");
    Value k = make_tensor(Tensor::randn(Shape{{1, 4, 8}}, opts), "k");
    Value v = make_tensor(Tensor::randn(Shape{{1, 4, 8}}, opts), "v");
    Value mask = make_tensor(Tensor::zeros(Shape{{1, 4, 4}}, opts), "mask"); // Assuming mask shape
    
    try {
        Value out = attention(q, k, v, mask);
        // Just check it runs and produces output of expected shape
        // Output should be [1, 4, 8] usually
        if (out.node->value.shape().dims != std::vector<int64_t>{1, 4, 8}) {
             std::cout << "❌ Unexpected output shape\n";
             return;
        }
        
        backward(sum(out));
        std::cout << "✅ Passed\n";
    } catch (const std::exception& e) {
        std::cout << "❌ Failed: " << e.what() << "\n";
    }
}

int main() {
    std::cout << "Running Attention Tests...\n";
    test_attention_op();
    return 0;
}
