#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>
#include "ad/ag_all.hpp"
#include "ad/optimizer/optim.hpp"
#include "ad/optimizer/loss_scaler.hpp"

using namespace ag;
using namespace OwnTensor;

#define ASSERT_NEAR(a, b, tol) \
    if (std::abs((a) - (b)) > (tol)) { \
        std::cerr << "Assertion Failed: " << (a) << " not near " << (b) << " (tol=" << (tol) << ")" << std::endl; \
        return false; \
    }

#define ASSERT_TRUE(cond, msg) \
    if (!(cond)) { \
        std::cerr << "Assertion Failed: " << msg << std::endl; \
        return false; \
    }

// Test 1: Adam Mixed Precision (BF16 params, FP32 moments/master)
bool test_adam_mixed_precision() {
    std::cout << "[Test 1] Adam Mixed Precision (BF16)... ";
    
    // Create BF16 parameter
    TensorOptions opts_bf16 = TensorOptions().with_dtype(Dtype::Bfloat16).with_req_grad(true);
    Tensor xt = Tensor::ones(Shape{{1, 1}}, opts_bf16);
    xt.fill(bfloat16_t(5.0f));
    
    Value x = make_tensor(xt, "x");
    
    // Adam should automatically create FP32 master weights and moments
    Adam optimizer({x}, 0.1f);
    
    // Run a few steps
    for (int i = 0; i < 10; ++i) {
        optimizer.zero_grad();
        Value y = x * x;
        backward(y);
        optimizer.step();
    }
    
    // Check if it converged towards 0
    float val = static_cast<float>(x.val().data<bfloat16_t>()[0]);
    ASSERT_TRUE(val < 5.0f, "Value should have decreased");
    
    std::cout << "PASS (Final Val: " << val << ")" << std::endl;
    return true;
}

// Test 2: Loss Scaler Basic Functionality
bool test_loss_scaler() {
    std::cout << "[Test 2] Loss Scaler Basic... ";
    
    LossScaler scaler(1024.0f);
    
    Tensor xt = Tensor::ones(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    xt.fill(1.0f);
    Value x = make_tensor(xt, "x");
    
    // 1. Scale loss
    Value loss = x * x;
    Value scaled_loss = scaler.scale_loss(loss);
    
    backward(scaled_loss);
    
    // Grad should be 2.0 * 1024.0 = 2048.0
    ASSERT_NEAR(x.grad().data<float>()[0], 2048.0f, 1e-3);
    
    // 2. Unscale gradients
    bool overflow = scaler.unscale_gradients({x});
    ASSERT_TRUE(!overflow, "Should not have overflowed");
    ASSERT_NEAR(x.grad().data<float>()[0], 2.0f, 1e-3);
    
    // 3. Update (no overflow)
    scaler.update(false);
    ASSERT_NEAR(scaler.scale(), 1024.0f, 1e-3); // Growth interval not reached
    
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 3: Loss Scaler Overflow Detection
bool test_loss_scaler_overflow() {
    std::cout << "[Test 3] Loss Scaler Overflow... ";
    
    LossScaler scaler(1024.0f);
    
    Tensor xt = Tensor::ones(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    Value x = make_tensor(xt, "x");
    
    // Manually set grad to INF
    x.grad() = Tensor::zeros(x.val().shape(), TensorOptions().with_dtype(x.val().dtype()).with_device(x.val().device()));
    x.grad().fill(std::numeric_limits<float>::infinity());
    
    bool overflow = scaler.unscale_gradients({x});
    ASSERT_TRUE(overflow, "Should have detected overflow");
    
    scaler.update(true);
    ASSERT_NEAR(scaler.scale(), 512.0f, 1e-3); // Should have backed off
    
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 4: SGD Mixed Precision
bool test_sgd_mixed_precision() {
    std::cout << "[Test 4] SGD Mixed Precision (BF16)... ";
    
    TensorOptions opts_bf16 = TensorOptions().with_dtype(Dtype::Bfloat16).with_req_grad(true);
    Tensor xt = Tensor::ones(Shape{{1, 1}}, opts_bf16);
    xt.fill(bfloat16_t(10.0f));
    Value x = make_tensor(xt, "x");
    
    SGDOptimizer optimizer({x}, 0.1f);
    
    for (int i = 0; i < 10; ++i) {
        optimizer.zero_grad();
        Value y = x * 2.0f;
        backward(y);
        optimizer.step();
    }
    
    // Grad is 2.0. Update is -0.1 * 2.0 = -0.2.
    // After 10 steps, x should be 10.0 - 2.0 = 8.0
    float val = static_cast<float>(x.val().data<bfloat16_t>()[0]);
    ASSERT_NEAR(val, 8.0f, 0.1f);
    
    std::cout << "PASS" << std::endl;
    return true;
}

int main() {
    std::cout << "===== Mixed Precision Training Tests (CPU) =====\n";
    
    bool all_pass = true;
    all_pass &= test_adam_mixed_precision();
    all_pass &= test_loss_scaler();
    all_pass &= test_loss_scaler_overflow();
    all_pass &= test_sgd_mixed_precision();
    
    if (all_pass) {
        std::cout << "\nRESULT: ALL TESTS PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "\nRESULT: SOME TESTS FAILED" << std::endl;
        return 1;
    }
}
