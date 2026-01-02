#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#include "ad/autodiff/autodiff.hpp"
#include "ad/ops/ops.hpp"

using namespace ag;

void test_layernorm() {
    std::cout << "[Test] LayerNorm... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    Value x = make_tensor(Tensor::randn(Shape{{2, 4}}, opts), "x");
    x.node->requires_grad_flag_ = true;
    x.node->grad = Tensor::zeros(Shape{{2, 4}}, opts);
    
    Value y = laynor(x);
    
    // Check output shape
    if (y.node->value.shape().dims != std::vector<int64_t>{2, 4}) {
        std::cout << "❌ Failed Forward Shape\n";
        return;
    }
    
    backward(sum(y));
    std::cout << "  Passed\n";
}

void test_rms() {
    std::cout << "[Test] RMSNorm... ";
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    Value x = make_tensor(Tensor::randn(Shape{{2, 4}}, opts), "x");
    
    Value y = rms(x);
    
    backward(sum(y));
    std::cout << "  Passed\n";
}

int main() {
    std::cout << "Running Normalization Tests...\n";
    test_layernorm();
    test_rms();
    return 0;
}
