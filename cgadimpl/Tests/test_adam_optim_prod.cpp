#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>
#include "ad/ag_all.hpp"
#include "optim.hpp"

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

// Test 1: Simple Convergence on Quadratic Function y = x^2
// Minimum is at x = 0.
bool test_convergence() {
    std::cout << "[Test 1] Convergence on y = x^2... ";
    
    Tensor xt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    xt.data<float>()[0] = 5.0f; // Start at x = 5
    Value x = make_tensor(xt, "x");
    
    Adam optimizer({x}, 0.05f); 
    
    for (int i = 0; i < 500; ++i) { // Increase steps to 500
        optimizer.zero_grad();
        Value y = x * x;
        backward(y);
        optimizer.step();
    }
    
    // After 500 steps with lr=0.05, x should be very close to 0
    ASSERT_NEAR(x.val().data<float>()[0], 0.0f, 0.05f);
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 2: Ensure zero_grad() works and no update happens when grad is 0
bool test_zero_grad_no_movement() {
    std::cout << "[Test 2] Zero Gradient / No Movement... ";
    
    Tensor xt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    xt.data<float>()[0] = 10.0f;
    Value x = make_tensor(xt, "x");
    
    Adam optimizer({x}, 0.1f);
    
    // First, run a step with 0 grad explicitly
    optimizer.zero_grad();
    optimizer.step();
    
    ASSERT_NEAR(x.val().data<float>()[0], 10.0f, 1e-7); // x should not have moved
    
    // Now set grad and zero it
    x.grad().data<float>()[0] = 1.0f;
    optimizer.zero_grad();
    ASSERT_NEAR(x.grad().data<float>()[0], 0.0f, 1e-7);
    
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 3: Mixed Parameters (some require grad, some don't)
bool test_requires_grad_filtering() {
    std::cout << "[Test 3] Requires Grad Filtering... ";
    
    Tensor x1t(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    x1t.data<float>()[0] = 1.0f;
    Value x1 = make_tensor(x1t, "x1");
    
    Tensor x2t(Shape{{1, 1}}, TensorOptions().with_req_grad(false));
    x2t.data<float>()[0] = 1.0f;
    Value x2 = make_tensor(x2t, "x2");
    
    Adam optimizer({x1, x2}, 0.1f);
    
    // Manually set gradients
    x1.grad().data<float>()[0] = 1.0f;
    // x2 won't even have a grad tensor allocated or used if requires_grad is false
    // but the optimizer should just skip it.
    
    optimizer.step();
    
    ASSERT_TRUE(x1.val().data<float>()[0] != 1.0f, "x1 should have updated");
    ASSERT_NEAR(x2.val().data<float>()[0], 1.0f, 1e-7); // x2 should NOT have updated
    
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 4: Numerical Stability with vanishing/exploding gradients
bool test_numerical_stability() {
    std::cout << "[Test 4] Numerical Stability (Exploding/Vanishing)... ";
    
    Tensor xt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    xt.data<float>()[0] = 1.0f;
    Value x = make_tensor(xt, "x");
    
    Adam optimizer({x}, 0.1f, 0.9f, 0.999f, 1e-8f);
    
    // Case A: Extremely small gradient (testing epsilon)
    x.grad().data<float>()[0] = 1e-15f;
    optimizer.step();
    ASSERT_TRUE(!std::isnan(x.val().data<float>()[0]), "Small grad caused NaN");
    
    // Case B: Extremely large gradient
    x.val().data<float>()[0] = 1.0f;
    x.grad().data<float>()[0] = 1e20f;
    optimizer.step();
    ASSERT_TRUE(!std::isnan(x.val().data<float>()[0]), "Large grad caused NaN");
    
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 5: Verify Bias Correction Logic
// Adam update with bias correction for step 1:
// m_1 = (1-b1) * g
// v_1 = (1-b2) * g^2
// m_hat = m_1 / (1-b1) = g
// v_hat = v_1 / (1-b2) = g^2
// update = alpha * m_hat / (sqrt(v_hat) + eps) = alpha * g / (sqrt(g^2) + eps) approx alpha * sign(g)
bool test_bias_correction() {
    std::cout << "[Test 5] Bias Correction Verification... ";
    
    float alpha = 0.1f;
    float g_val = 2.0f;
    
    Tensor xt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    xt.data<float>()[0] = 10.0f;
    Value x = make_tensor(xt, "x");
    
    Adam optimizer({x}, alpha);
    
    x.grad().data<float>()[0] = g_val;
    optimizer.step();
    
    // Step 1 expected update if bias correction is correct:
    // m_hat = 2.0, v_hat = 4.0
    // update = 0.1 * 2 / (sqrt(4) + 1e-8) = 0.1 * 2 / 2 = 0.1
    // new_x = 10 - 0.1 = 9.9
    ASSERT_NEAR(x.val().data<float>()[0], 9.9f, 1e-4);
    
    std::cout << "PASS" << std::endl;
    return true;
}

int main() {
    std::cout << "===== Adam Optimizer Production-Grade Edge Case Tests =====\n";
    
    bool all_pass = true;
    all_pass &= test_convergence();
    all_pass &= test_zero_grad_no_movement();
    all_pass &= test_requires_grad_filtering();
    all_pass &= test_numerical_stability();
    all_pass &= test_bias_correction();
    
    std::cout << "\n===========================================================\n";
    if (all_pass) {
        std::cout << "RESULT: ALL TESTS PASSED ✅" << std::endl;
        return 0;
    } else {
        std::cout << "RESULT: SOME TESTS FAILED ❌" << std::endl;
        return 1;
    }
}
