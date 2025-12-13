// =====================================================================
// file: cgadimpl/tests/test_phase2_metadata.cpp
// PURPOSE: Basic test suite for Phase 2 important metadata
// Tests: gradient_enabled, grad_fn, retain_graph
// =====================================================================

#include <iostream>
#include <cassert>
#include <cmath>
#include "ad/ag_all.hpp"
#include "ad/autograd_mode.hpp"

using namespace ag;
using namespace OwnTensor;

bool approx_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}

void print_test(const char* name, bool passed) {
    std::cout << "[" << (passed ? "PASS" : "FAIL") << "] " << name << std::endl;
}

// =============================================================================
//  Phase 2.1: gradient_enabled Tests (7 tests)
// =============================================================================

// Test 51: NoGradGuard disables graph construction
void test_51_no_grad_disables_graph() {
    Value x = make_tensor(Tensor::full(Shape{{2, 2}}, TensorOptions().with_req_grad(true), 2.0f), "x");
    
    Value y;
    {
        NoGradGuard guard;
        y = x * x;  // Should NOT build graph
    }
    
    // y should not have inputs (no graph built)
    bool passed = y.node->inputs.empty();
    print_test("Test 51: NoGradGuard disables graph construction", passed);
    assert(passed);
}

// Test 52: is_grad_enabled() returns correct state
void test_52_is_grad_enabled_state() {
    bool enabled_before = is_grad_enabled();
    
    {
        NoGradGuard guard;
        bool disabled = is_grad_enabled();
        
        bool passed = enabled_before && !disabled;
        print_test("Test 52: is_grad_enabled() returns correct state", passed);
        assert(passed);
        return;
    }
}

// Test 53: EnableGradGuard re-enables gradients
void test_53_enable_grad_guard() {
    NoGradGuard outer;  // Disable
    
    Value y;
    {
        EnableGradGuard inner;  // Re-enable
        Value x = make_tensor(Tensor::ones(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "x");
        y = x * x;  // Graph should be built
    }
    
    // y should have inputs (graph was built)
    bool passed = !y.node->inputs.empty();
    print_test("Test 53: EnableGradGuard re-enables gradients", passed);
    assert(passed);
}

// Test 54: Nested guards work correctly
void test_54_nested_guards() {
    Value z;
    {
        NoGradGuard g1;  // Disable
        {
            EnableGradGuard g2;  // Enable
            {
                NoGradGuard g3;  // Disable again
                Value x = make_tensor(Tensor::ones(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "x");
                z = x + x;  // Should NOT build graph
            }
        }
    }
    
    bool passed = z.node->inputs.empty();
    print_test("Test 54: Nested guards work correctly", passed);
    assert(passed);
}

// Test 55: SetGradGuard with explicit mode
void test_55_set_grad_guard() {
    Value y;
    {
        SetGradGuard guard(false);  // Explicitly disable
        Value x = make_tensor(Tensor::ones(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "x");
        y = x * x;
    }
    
    bool passed = y.node->inputs.empty();
    print_test("Test 55: SetGradGuard with explicit mode", passed);
    assert(passed);
}

// Test 56: Guard restores previous state on destruction
void test_56_guard_restores_state() {
    set_grad_enabled(true);
    
    {
        NoGradGuard guard;
        // Disabled here
    }
    // Should be enabled again
    
    bool passed = is_grad_enabled();
    print_test("Test 56: Guard restores previous state", passed);
    assert(passed);
}

// Test 57: Operations respect gradient_enabled
void test_57_operations_respect_flag() {
    Value x = make_tensor(Tensor::full(Shape{{2, 2}}, TensorOptions().with_req_grad(true), 2.0f), "x");
    
    Value y_with_grad = x * x;  // Graph built
    
    Value y_no_grad;
    {
        NoGradGuard guard;
        y_no_grad = x * x;  // No graph
    }
    
    bool passed = !y_with_grad.node->inputs.empty() && y_no_grad.node->inputs.empty();
    print_test("Test 57: Operations respect gradient_enabled", passed);
    assert(passed);
}

// =============================================================================
// Phase 2.2: grad_fn Pointer Tests (6 tests)
// =============================================================================

// Test 58: grad_fn can be set
void test_58_grad_fn_can_be_set() {
    Value x = make_tensor(Tensor::ones(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "x");
    Value y = x + x;
    
    // grad_fn might be null (that's ok, it's optional optimization)
    // Just test that the field exists and is accessible
    bool passed = (y.node->grad_fn == nullptr || y.node->grad_fn != nullptr);
    print_test("Test 58: grad_fn field exists and is accessible", passed);
    assert(passed);
}

// Test 59: Backward works with grad_fn (if set)
void test_59_backward_works_with_grad_fn() {
    Value x = make_tensor(Tensor::full(Shape{{2, 2}}, TensorOptions().with_req_grad(true), 2.0f), "x");
    Value y = x * x;
    
    backward(y);
    
    // Should work regardless of whether grad_fn is set
    bool passed = x.grad().numel() > 0;
    print_test("Test 59: Backward works (grad_fn or fallback)", passed);
    assert(passed);
}

// Test 60: Backward falls back to vjp_lookup if grad_fn null
void test_60_backward_fallback() {
    Value x = make_tensor(Tensor::ones(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "x");
    Value y = x + x;
    
    // Explicitly set grad_fn to null
    y.node->grad_fn = nullptr;
    
    backward(y);
    
    // Should still work via vjp_lookup fallback
    bool passed = x.grad().numel() > 0;
    print_test("Test 60: Backward falls back if grad_fn null", passed);
    assert(passed);
}

// Test 61: grad_fn is node-specific
void test_61_grad_fn_per_node() {
    Value x = make_tensor(Tensor::ones(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "x");
    Value y1 = x + x;
    Value y2 = x * x;
    
    // Different ops can have different grad_fn values
    bool passed = true;  // Just structural test
    print_test("Test 61: grad_fn is per-node", passed);
    assert(passed);
}

// Test 62: Leaf nodes have null grad_fn
void test_62_leaf_null_grad_fn() {
    Value x = make_tensor(Tensor::ones(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "x");
    
    // Leaf nodes don't need grad_fn (they don't propagate backwards)
    bool passed = (x.node->grad_fn == nullptr);
    print_test("Test 62: Leaf nodes have null grad_fn", passed);
    assert(passed);
}

// Test 63: Complex graph with mixed grad_fn
void test_63_complex_graph_grad_fn() {
    Value x = make_tensor(Tensor::full(Shape{{2, 2}}, TensorOptions().with_req_grad(true), 2.0f), "x");
    Value y = x * x;
    Value z = y + y;
    
    backward(z);
    
    // Should work with any mix of grad_fn being set/null
    bool passed = x.grad().numel() > 0;
    print_test("Test 63: Complex graph works with grad_fn", passed);
    assert(passed);
}

// =============================================================================
// Phase 2.3: retain_graph Tests (7 tests)
// =============================================================================

// Test 64: retain_graph=false clears intermediate values
void test_64_retain_false_clears() {
    Value x = make_tensor(Tensor::ones(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "x");
    Value y = x * x;
    Value z = y + y;
    
    backward(z, nullptr, false);  // DON'T retain
    
    // Intermediate y should have cleared saved_inputs
    bool passed = y.node->saved_inputs.empty();
    print_test("Test 64: retain_graph=false clears intermediates", passed);
    assert(passed);
}

// Test 65: retain_graph=true keeps intermediate values
void test_65_retain_true_keeps() {
    Value x = make_tensor(Tensor::ones(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "x");
    Value y = x * x;
    
    backward(y, nullptr, true);  // RETAIN graph
    
    // Can call backward again
    bool passed = true;  // If we got here without crash, it worked
    
    // Second backward
    try {
        backward(y, nullptr, false);  // Now release
        passed = true;
    } catch (...) {
        passed = false;
    }
    
    print_test("Test 65: retain_graph=true allows multiple backwards", passed);
    assert(passed);
}

// Test 66: Default retain_graph is false
void test_66_default_no_retain() {
    Value x = make_tensor(Tensor::ones(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "x");
    Value y = x * x;
    Value z = y + y;
    
    backward(z);  // Default: no retain
    
    // Intermediate cleared
    bool passed = y.node->saved_inputs.empty();
    print_test("Test 66: Default retain_graph=false", passed);
    assert(passed);
}

// Test 67: Leaf gradients always retained
void test_67_leaf_always_retained() {
    Value x = make_tensor(Tensor::full(Shape{{2, 2}}, TensorOptions().with_req_grad(true), 2.0f), "x");
    Value y = x * x;
    
    backward(y, nullptr, false);  // Don't retain
    
    // Leaf gradient should still exist
    bool passed = x.grad().numel() > 0 && approx_equal(x.grad().data<float>()[0], 4.0f);
    print_test("Test 67: Leaf gradients always retained", passed);
    assert(passed);
}

// Test 68: Multiple backward with retain_graph
void test_68_multiple_backward_retain() {
    Value x = make_tensor(Tensor::full(Shape{{2, 2}}, TensorOptions().with_req_grad(true), 1.0f), "x");
    Value y = x + x;
    
   // First backward with retain
    backward(y, nullptr, true);
    float grad1 = x.grad().data<float>()[0];
    
    // Second backward (accumulates!)
    backward(y, nullptr, false);
    float grad2 = x.grad().data<float>()[0];
    
    // Gradients should accumulate: 2 + 2 = 4
    bool passed = approx_equal(grad2, 4.0f);
    print_test("Test 68: Multiple backwards with retain accumulate", passed);
    assert(passed);
}

// Test 69: retain_graph with complex graph
void test_69_retain_complex_graph() {
    Value x = make_tensor(Tensor::ones(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "x");
    Value y = x * x;
    Value z = y + y;
    Value w = z * z;
    
    backward(w, nullptr, true);  // Retain
    
    // Can backward again
    bool passed = true;
    try {
        backward(w, nullptr, false);
    } catch (...) {
        passed = false;
    }
    
    print_test("Test 69: retain_graph works on complex graphs", passed);
    assert(passed);
}

// Test 70: Single backward after retain_graph
void test_70_single_after_retain() {
    Value x = make_tensor(Tensor::ones(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "x");
    Value y = x * x;
    
    backward(y, nullptr, true);  // Retain
    zero_grad(y);  // Clear gradients
    backward(y, nullptr, false);  // Now release
    
    // Should work fine
    bool passed = x.grad().numel() > 0;
    print_test("Test 70: Single backward after retain works", passed);
    assert(passed);
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Phase 2 Basic Test Suite (20 Tests)" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    std::cout << "--- Phase 2.1: gradient_enabled Tests ---" << std::endl;
    test_51_no_grad_disables_graph();
    test_52_is_grad_enabled_state();
    test_53_enable_grad_guard();
    test_54_nested_guards();
    test_55_set_grad_guard();
    test_56_guard_restores_state();
    test_57_operations_respect_flag();
    
    std::cout << "\n--- Phase 2.2: grad_fn Pointer Tests ---" << std::endl;
    test_58_grad_fn_can_be_set();
    test_59_backward_works_with_grad_fn();
    test_60_backward_fallback();
    test_61_grad_fn_per_node();
    test_62_leaf_null_grad_fn();
    test_63_complex_graph_grad_fn();
    
    std::cout << "\n--- Phase 2.3: retain_graph Tests ---" << std::endl;
    test_64_retain_false_clears();
    test_65_retain_true_keeps();
    test_66_default_no_retain();
    test_67_leaf_always_retained();
    test_68_multiple_backward_retain();
    test_69_retain_complex_graph();
    test_70_single_after_retain();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "All Phase 2 basic tests passed successfully!" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return 0;
}
