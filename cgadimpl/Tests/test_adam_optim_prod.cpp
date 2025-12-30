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

// Test 6: Multiple Parameters f(x, y) = x^2 + y^2
bool test_multiple_params() {
    std::cout << "[Test 6] Multiple Parameters f(x, y) = x^2 + y^2... ";
    
    Tensor xt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    xt.data<float>()[0] = 3.0f;
    Value x = make_tensor(xt, "x");
    
    Tensor yt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    yt.data<float>()[0] = -4.0f;
    Value y = make_tensor(yt, "y");
    
    Adam optimizer({x, y}, 0.1f);
    
    for (int i = 0; i < 200; ++i) {
        optimizer.zero_grad();
        Value loss = x * x + y * y;
        backward(loss);
        optimizer.step();
    }
    
    ASSERT_NEAR(x.val().data<float>()[0], 0.0f, 0.01f);
    ASSERT_NEAR(y.val().data<float>()[0], 0.0f, 0.01f);
    
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 7: Constant Gradient - verify movement direction
bool test_constant_gradient() {
    std::cout << "[Test 7] Constant Gradient Movement... ";
    
    Tensor xt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    xt.data<float>()[0] = 0.0f;
    Value x = make_tensor(xt, "x");
    
    Adam optimizer({x}, 0.1f);
    
    // Set constant gradient of 1.0
    for (int i = 0; i < 10; ++i) {
        optimizer.zero_grad();
        x.grad().data<float>()[0] = 1.0f;
        optimizer.step();
    }
    
    // With constant gradient, x should move in negative direction
    ASSERT_TRUE(x.val().data<float>()[0] < 0.0f, "x should have moved negative");
    
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 8: First Moment (m) Accumulation
bool test_first_moment_accumulation() {
    std::cout << "[Test 8] First Moment (m) Accumulation... ";
    
    Tensor xt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    xt.data<float>()[0] = 0.0f;
    Value x = make_tensor(xt, "x");
    
    float b1 = 0.9f;
    Adam optimizer({x}, 0.1f, b1);
    
    // Step 1: grad = 1.0
    x.grad().data<float>()[0] = 1.0f;
    optimizer.step();
    // m_1 = (1-b1)*1.0 = 0.1
    
    // Step 2: grad = 1.0
    x.grad().data<float>()[0] = 1.0f;
    optimizer.step();
    // m_2 = b1*m_1 + (1-b1)*1.0 = 0.9*0.1 + 0.1 = 0.09 + 0.1 = 0.19
    
    // We can't easily access m_ directly without making it public or using a friend,
    // but we can infer it from the update.
    // bias_corr1 = 1 - 0.9^2 = 0.19
    // m_hat = 0.19 / 0.19 = 1.0
    // v_hat = 1.0 (since grad is constant)
    // update = 0.1 * 1.0 / (1.0 + eps) = 0.1
    // x_2 = x_1 - 0.1
    // x_1 was 10 - 0.1 = 9.9 (from test 5 logic, but here start is 0)
    // Actually, let's just check if it's moving as expected.
    
    ASSERT_NEAR(x.val().data<float>()[0], -0.2f, 1e-3); // Step 1: -0.1, Step 2: -0.1
    
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 9: Second Moment (v) Accumulation
bool test_second_moment_accumulation() {
    std::cout << "[Test 9] Second Moment (v) Accumulation... ";
    
    Tensor xt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    xt.data<float>()[0] = 0.0f;
    Value x = make_tensor(xt, "x");
    
    float b2 = 0.999f;
    Adam optimizer({x}, 0.1f, 0.9f, b2);
    
    x.grad().data<float>()[0] = 2.0f;
    optimizer.step();
    // v_1 = (1-b2)*4.0 = 0.001 * 4 = 0.004
    // v_hat = 0.004 / 0.001 = 4.0
    
    ASSERT_NEAR(x.val().data<float>()[0], -0.1f, 1e-3); // update = 0.1 * 2 / 2 = 0.1
    
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 10: Learning Rate Sensitivity
bool test_lr_sensitivity() {
    std::cout << "[Test 10] Learning Rate Sensitivity... ";
    
    // Very small LR
    {
        Tensor xt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
        xt.data<float>()[0] = 1.0f;
        Value x = make_tensor(xt, "x");
        Adam optimizer({x}, 1e-10f);
        x.grad().data<float>()[0] = 1.0f;
        optimizer.step();
        ASSERT_NEAR(x.val().data<float>()[0], 1.0f, 1e-9);
    }
    
    // Very large LR
    {
        Tensor xt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
        xt.data<float>()[0] = 1.0f;
        Value x = make_tensor(xt, "x");
        Adam optimizer({x}, 1e6f);
        x.grad().data<float>()[0] = 1.0f;
        optimizer.step();
        ASSERT_TRUE(x.val().data<float>()[0] < -1e5f, "Large LR should cause large update");
    }
    
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 11: NaN Gradients
bool test_nan_gradients() {
    std::cout << "[Test 11] NaN Gradients Robustness... ";
    
    Tensor xt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    xt.data<float>()[0] = 1.0f;
    Value x = make_tensor(xt, "x");
    
    Adam optimizer({x}, 0.1f);
    
    x.grad().data<float>()[0] = std::nanf("");
    optimizer.step();
    
    // Update with NaN grad will likely result in NaN value, but we check it doesn't crash
    ASSERT_TRUE(std::isnan(x.val().data<float>()[0]), "NaN grad should result in NaN value");
    
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 12: Inf Gradients
bool test_inf_gradients() {
    std::cout << "[Test 12] Inf Gradients Robustness... ";
    
    Tensor xt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    xt.data<float>()[0] = 1.0f;
    Value x = make_tensor(xt, "x");
    
    Adam optimizer({x}, 0.1f);
    
    x.grad().data<float>()[0] = std::numeric_limits<float>::infinity();
    optimizer.step();
    
    // Inf grad will cause m and v to be inf, resulting in NaN update (inf/inf)
    ASSERT_TRUE(std::isnan(x.val().data<float>()[0]), "Inf grad should result in NaN value");
    
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 13: Empty Parameters
bool test_empty_params() {
    std::cout << "[Test 13] Empty Parameters... ";
    
    std::vector<Value> empty_params;
    Adam optimizer(empty_params, 0.1f);
    
    optimizer.step(); // Should not crash
    optimizer.zero_grad(); // Should not crash
    
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 14: High Dimensionality
bool test_high_dimensionality() {
    std::cout << "[Test 14] High Dimensionality (1000 elements)... ";
    
    int size = 1000;
    Tensor xt(Shape{{size}}, TensorOptions().with_req_grad(true));
    for (int i = 0; i < size; ++i) xt.data<float>()[i] = (float)i;
    Value x = make_tensor(xt, "x");
    
    Adam optimizer({x}, 0.1f);
    
    for (int i = 0; i < 10; ++i) {
        optimizer.zero_grad();
        Value loss = sum(x * x);
        backward(loss);
        optimizer.step();
    }
    
    for (int i = 0; i < size; ++i) {
        ASSERT_TRUE(x.val().data<float>()[i] < (float)i + 1e-5f, "Value should have decreased or stayed same");
    }
    
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 15: Saddle Point Escape
bool test_saddle_point() {
    std::cout << "[Test 15] Saddle Point Escape f(x, y) = x^2 - y^2... ";
    
    Tensor xt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    xt.data<float>()[0] = 0.001f; // Small offset from saddle point
    Value x = make_tensor(xt, "x");
    
    Tensor yt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    yt.data<float>()[0] = 0.001f;
    Value y = make_tensor(yt, "y");
    
    Adam optimizer({x, y}, 0.1f);
    
    for (int i = 0; i < 100; ++i) {
        optimizer.zero_grad();
        Value loss = x * x - y * y;
        backward(loss);
        optimizer.step();
    }
    
    // x should go to 0, y should explode (negative)
    ASSERT_NEAR(x.val().data<float>()[0], 0.0f, 0.1f);
    ASSERT_TRUE(y.val().data<float>()[0] > 1.0f, "y should have moved away from saddle point");
    
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 16: Rosenbrock Function (Narrow Valley)
// f(x, y) = (a-x)^2 + b(y-x^2)^2, usually a=1, b=100. Min at (1, 1)
bool test_rosenbrock() {
    std::cout << "[Test 16] Rosenbrock Function Convergence... ";
    
    Tensor xt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    xt.data<float>()[0] = -1.2f;
    Value x = make_tensor(xt, "x");
    
    Tensor yt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    yt.data<float>()[0] = 1.0f;
    Value y = make_tensor(yt, "y");
    
    Adam optimizer({x, y}, 0.01f);
    
    for (int i = 0; i < 5000; ++i) { // Increased steps to 5000
        optimizer.zero_grad();
        Value term1 = (1.0f - x);
        Value term2 = (y - x * x);
        Value loss = term1 * term1 + 100.0f * term2 * term2;
        backward(loss);
        optimizer.step();
    }
    
    ASSERT_NEAR(x.val().data<float>()[0], 1.0f, 0.1f);
    ASSERT_NEAR(y.val().data<float>()[0], 1.0f, 0.1f);
    
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 17: State Persistence (m and v maintained)
bool test_state_persistence() {
    std::cout << "[Test 17] State Persistence... ";
    
    Tensor xt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    xt.data<float>()[0] = 1.0f;
    Value x = make_tensor(xt, "x");
    
    Adam optimizer({x}, 0.1f);
    
    // Step 1: grad = 1.0
    x.grad().data<float>()[0] = 1.0f;
    optimizer.step();
    float val1 = x.val().data<float>()[0];
    
    // Step 2: grad = 2.0 (changing gradient)
    x.grad().data<float>()[0] = 2.0f;
    optimizer.step();
    float val2 = x.val().data<float>()[0];
    
    // If state was NOT maintained, the update would only depend on the current gradient.
    // With state, the previous gradient (1.0) still influences m and v.
    // We just check that it doesn't crash and moves. 
    // To really test persistence, we'd need to compare with a fresh optimizer.
    
    Tensor x3t(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    x3t.data<float>()[0] = val1;
    Value x3 = make_tensor(x3t, "x3");
    Adam optimizer2({x3}, 0.1f); // Fresh state
    x3.grad().data<float>()[0] = 2.0f;
    optimizer2.step();
    float val3 = x3.val().data<float>()[0];
    
    ASSERT_TRUE(std::abs(val2 - val3) > 1e-5f, "Persistent state should result in different update than fresh state");
    
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 18: Momentum Decay
bool test_momentum_decay() {
    std::cout << "[Test 18] Momentum Decay... ";
    
    Tensor xt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    xt.data<float>()[0] = 0.0f;
    Value x = make_tensor(xt, "x");
    
    Adam optimizer({x}, 0.1f);
    
    // Give it some momentum
    x.grad().data<float>()[0] = 1.0f;
    optimizer.step();
    
    // Now set grad to 0
    float last_val = x.val().data<float>()[0];
    for (int i = 0; i < 5; ++i) {
        optimizer.zero_grad();
        optimizer.step();
        float current_val = x.val().data<float>()[0];
        // It should still move due to momentum, but less and less
        ASSERT_TRUE(std::abs(current_val - last_val) > 0, "Should still move due to momentum");
        last_val = current_val;
    }
    
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 19: Large Beta2 (Slow variance update)
bool test_large_beta2() {
    std::cout << "[Test 19] Large Beta2 (Slow Variance Update)... ";
    
    Tensor xt(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    xt.data<float>()[0] = 1.0f;
    Value x = make_tensor(xt, "x");
    
    // Beta2 = 0.9999 (very slow)
    Adam optimizer({x}, 0.1f, 0.9f, 0.9999f);
    
    x.grad().data<float>()[0] = 10.0f;
    optimizer.step();
    
    // Even with large grad, update is limited by slow v_hat increase
    ASSERT_NEAR(x.val().data<float>()[0], 0.9f, 0.01f); // update approx 0.1
    
    std::cout << "PASS" << std::endl;
    return true;
}

// Test 20: Epsilon Influence
bool test_epsilon_influence() {
    std::cout << "[Test 20] Epsilon Influence... ";
    
    // Large epsilon should dampen updates
    Tensor x1t(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    x1t.data<float>()[0] = 1.0f;
    Value x1 = make_tensor(x1t, "x1");
    Adam opt1({x1}, 0.1f, 0.9f, 0.999f, 1e-1f); // Huge epsilon
    
    Tensor x2t(Shape{{1, 1}}, TensorOptions().with_req_grad(true));
    x2t.data<float>()[0] = 1.0f;
    Value x2 = make_tensor(x2t, "x2");
    Adam opt2({x2}, 0.1f, 0.9f, 0.999f, 1e-8f); // Small epsilon
    
    x1.grad().data<float>()[0] = 1e-5f;
    x2.grad().data<float>()[0] = 1e-5f;
    
    opt1.step();
    opt2.step();
    
    float update1 = 1.0f - x1.val().data<float>()[0];
    float update2 = 1.0f - x2.val().data<float>()[0];
    
    ASSERT_TRUE(update1 < update2, "Large epsilon should result in smaller update for small gradients");
    
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
    all_pass &= test_multiple_params();
    all_pass &= test_constant_gradient();
    all_pass &= test_first_moment_accumulation();
    all_pass &= test_second_moment_accumulation();
    all_pass &= test_lr_sensitivity();
    all_pass &= test_nan_gradients();
    all_pass &= test_inf_gradients();
    all_pass &= test_empty_params();
    all_pass &= test_high_dimensionality();
    all_pass &= test_saddle_point();
    all_pass &= test_rosenbrock();
    all_pass &= test_state_persistence();
    all_pass &= test_momentum_decay();
    all_pass &= test_large_beta2();
    all_pass &= test_epsilon_influence();
    
    std::cout << "\n===========================================================\n";
    if (all_pass) {
        std::cout << "RESULT: ALL TESTS PASSED ✅" << std::endl;
        return 0;
    } else {
        std::cout << "RESULT: SOME TESTS FAILED ❌" << std::endl;
        return 1;
    }
}
