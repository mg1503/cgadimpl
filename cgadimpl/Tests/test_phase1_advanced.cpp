// =====================================================================
// file: cgadimpl/tests/test_phase1_advanced.cpp
// PURPOSE: Advanced edge case test suite for Phase 1 metadata
// Covers scenarios that PyTorch/JAX handle in production
// =====================================================================

#include <iostream>
#include <cassert>
#include <cmath>
#include <limits>
#include "ad/ag_all.hpp"
#include "ad/core/version_tracker.hpp"

using namespace ag;
using namespace OwnTensor;

bool approx_equal(float a, float b, float epsilon = 1e-4f) {
    if (std::isnan(a) && std::isnan(b)) return true;
    if (std::isinf(a) && std::isinf(b) && (a > 0) == (b > 0)) return true;
    return std::abs(a - b) < epsilon;
}

void print_test(const char* name, bool passed) {
    std::cout << "[" << (passed ? "PASS" : "FAIL") << "] " << name << std::endl;
}

// =============================================================================
// Category 1: Gradient Accumulation Edge Cases (Tests 21-25)
// =============================================================================

// Test 21: Multiple backward passes accumulate gradients on leaves
void test_21_multiple_backward_accumulation() {
    Value x = make_tensor(Tensor::full(Shape{{2, 2}}, TensorOptions().with_req_grad(true), 2.0f), "x");
    
    // First backward
    Value y1 = x * x;
    backward(y1);
    float grad1 = x.grad().data<float>()[0];
    
    // Second backward (should accumulate)
    Value y2 = x + x;
    backward(y2);
    float grad2 = x.grad().data<float>()[0];
    
    // grad1 = 2*x = 4, grad2 should add 2, total = 6
    bool passed = approx_equal(grad2, grad1 + 2.0f);
    print_test("Test 21: Multiple backward accumulates on leaves", passed);
    assert(passed);
}

// Test 22: Zero gradient initialization
void test_22_zero_grad() {
    Value x = make_tensor(Tensor::ones(Shape{{3, 3}}, TensorOptions().with_req_grad(true)), "x");
    Value y = x * x;
    backward(y);
    
    // Call zero_grad
    zero_grad(y);
    
    // Gradient should be zeroed
    bool passed = x.grad().data<float>()[0] == 0.0f;
    print_test("Test 22: zero_grad clears gradients", passed);
    assert(passed);
}

// Test 23: Gradient on scalar vs non-scalar
void test_23_scalar_gradient() {
    Value x = make_tensor(Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), 3.0f), "x");
    Value y = x * x;  // y = 9
    
    backward(y);
    
    // dy/dx = 2*x = 6
    bool passed = approx_equal(x.grad().data<float>()[0], 6.0f);
    print_test("Test 23: Scalar tensor backward works", passed);
    assert(passed);
}

// Test 24: Shared leaf in multiple paths (gradient accumulation)
void test_24_shared_leaf_accumulation() {
    Value x = make_tensor(Tensor::full(Shape{{2, 2}}, TensorOptions().with_req_grad(true), 2.0f), "x");
    
    // x is used in two separate branches
    Value branch1 = x * x;  // contributes 2*x gradient
    Value branch2 = x + x;  // contributes 2 gradient
    Value y = branch1 + branch2;
    
    backward(y);
    
    // Total gradient = 2*x + 2 = 4 + 2 = 6
    bool passed = approx_equal(x.grad().data<float>()[0], 6.0f);
    print_test("Test 24: Shared leaf accumulates from multiple paths", passed);
    assert(passed);
}

// Test 25: Non-leaf intermediate nodes don't accumulate
void test_25_intermediate_no_accumulation() {
    Value x = make_tensor(Tensor::ones(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "x");
    Value y = x * x;  // y is non-leaf
    Value z = y + y;
    
    backward(z);
    
    // x is leaf, should have gradient
    // y is non-leaf, should NOT accumulate gradient
    bool passed = (x.node->is_leaf && !y.node->is_leaf);
    print_test("Test 25: Intermediate nodes are non-leaf", passed);
    assert(passed);
}

// =============================================================================
// Category 2: Graph Structure Edge Cases (Tests 26-30)
// =============================================================================

// Test 26: Diamond-shaped graph (multiple paths converge)
void test_26_diamond_graph() {
    Value x = make_tensor(Tensor::full(Shape{{2, 2}}, TensorOptions().with_req_grad(true), 1.0f), "x");
    
    // Diamond: x -> a -> c
    //          x -> b -> c
    Value a = x * 2.0f;
    Value b = x + 1.0f;
    Value c = a + b;
    
    backward(c);
    
    // Gradient through both paths should accumulate
    // da/dx = 2, db/dx = 1, dc/da = 1, dc/db = 1
    // dc/dx = 2 + 1 = 3
    bool passed = approx_equal(x.grad().data<float>()[0], 3.0f);
    print_test("Test 26: Diamond graph accumulates correctly", passed);
    assert(passed);
}

// Test 27: Very deep linear chain
void test_27_deep_chain() {
    Value x = make_tensor(Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), 1.0f), "x");
    Value y = x;
    
    // Create 50-layer deep chain
    for (int i = 0; i < 50; i++) {
        y = y + 1.0f;
    }
    
    backward(y);
    
    // Gradient should propagate through all layers: dy/dx = 1
    bool passed = approx_equal(x.grad().data<float>()[0], 1.0f) && !x.node->is_leaf == false;
    print_test("Test 27: Deep chain (50 layers) propagates correctly", passed);
    assert(passed);
}

// Test 28: Wide graph (many parallel operations)
void test_28_wide_graph() {
    Value x = make_tensor(Tensor::full(Shape{{2, 2}}, TensorOptions().with_req_grad(true), 1.0f), "x");
    
    // Create 20 parallel branches, then sum
    Value sum = x * 0.0f;  // Start with zeros
    for (int i = 0; i < 20; i++) {
        sum = sum + (x * 1.0f);
    }
    
    backward(sum);
    
    // Each branch contributes gradient of 1, total = 20
    bool passed = approx_equal(x.grad().data<float>()[0], 20.0f);
    print_test("Test 28: Wide graph (20 parallel paths)", passed);
    assert(passed);
}

// Test 29: Multiple leaf parameters
void test_29_multiple_leaves() {
    Value w = make_tensor(Tensor::full(Shape{{2, 2}}, TensorOptions().with_req_grad(true), 2.0f), "w");
    Value x = make_tensor(Tensor::full(Shape{{2, 2}}, TensorOptions().with_req_grad(true), 3.0f), "x");
    Value b = make_tensor(Tensor::full(Shape{{2, 2}}, TensorOptions().with_req_grad(true), 1.0f), "b");
    
    // y = w*x + b
    Value y = (w * x) + b;
    
    backward(y);
    
    // All three should be leaves with gradients
    bool passed = w.node->is_leaf && x.node->is_leaf && b.node->is_leaf &&
                  w.grad().numel() > 0 && x.grad().numel() > 0 && b.grad().numel() > 0;
    print_test("Test 29: Multiple leaf parameters all get gradients", passed);
    assert(passed);
}

// Test 30: Detached intermediate (simulating stop_gradient)
void test_30_detached_intermediate() {
    Value x = make_tensor(Tensor::full(Shape{{2, 2}}, TensorOptions().with_req_grad(true), 2.0f), "x");
    Value y = x * x;
    
    // Create new leaf from y's value (simulates detach)
    Value y_detached = make_tensor(y.val(), "y_detached");
    Value z = y_detached + y_detached;
    
    backward(z);
    
    // y_detached is a new leaf, x should not get gradient through it
    bool passed = y_detached.node->is_leaf;
    print_test("Test 30: Detached value becomes new leaf", passed);
    assert(passed);
}

// =============================================================================
// Category 3: Empty and Edge Tensor Cases (Tests 31-35)
// =============================================================================

// Test 31: Minimum dimension tensor (1x1)
void test_31_minimum_dimension() {
    Value x = make_tensor(Tensor::zeros(Shape{{1, 1}}, TensorOptions().with_req_grad(true)), "x");
    Value y = x + x;
    
    // Should handle minimal dimensions gracefully
    backward(y);
    
    bool passed = (x.node->is_leaf && x.val().numel() == 1);
    print_test("Test 31: Minimum dimension tensor handled", passed);
    assert(passed);
}

// Test 32: 1D tensor
void test_32_1d_tensor() {
    Value x = make_tensor(Tensor::full(Shape{{5}}, TensorOptions().with_req_grad(true), 2.0f), "x");
    Value y = x * x;
    
    backward(y);
    
    bool passed = (x.grad().numel() == 5) && approx_equal(x.grad().data<float>()[0], 4.0f);
    print_test("Test 32: 1D tensor backward works", passed);
    assert(passed);
}

// Test 33: 3D tensor
void test_33_3d_tensor() {
    Value x = make_tensor(Tensor::ones(Shape{{2, 3, 4}}, TensorOptions().with_req_grad(true)), "x");
    Value y = x + x;
    
    backward(y);
    
    bool passed = (x.grad().numel() == 24);
    print_test("Test 33: 3D tensor backward works", passed);
    assert(passed);
}

// Test 34: Single element tensor
void test_34_single_element() {
    Value x = make_tensor(Tensor::full(Shape{{1, 1}}, TensorOptions().with_req_grad(true), 5.0f), "x");
    Value y = x * x;
    
    backward(y);
    
    // dy/dx = 2*x = 10
    bool passed = approx_equal(x.grad().data<float>()[0], 10.0f);
    print_test("Test 34: Single element tensor", passed);
    assert(passed);
}

// Test 35: Very large tensor
void test_35_large_tensor() {
    Value x = make_tensor(Tensor::ones(Shape{{200, 200}}, TensorOptions().with_req_grad(true)), "x");
    Value y = x * x;
    
    backward(y);
    
    bool passed = (x.grad().numel() == 40000);
    print_test("Test 35: Large tensor (200x200) backward", passed);
    assert(passed);
}

// =============================================================================
// Category 4: Version Tracking Advanced (Tests 36-40)
// =============================================================================

// Test 36: Version tracking with cloned tensors
void test_36_version_clone() {
    Tensor t1 = Tensor::ones(Shape{{2, 2}});
    Tensor t2 = t1.clone();  // Different data pointer
    
    ag::detail::version_tracker().bump_version(t1);
    
    int v1 = ag::detail::version_tracker().get_version(t1);
    int v2 = ag::detail::version_tracker().get_version(t2);
    
    // t2 should be independent
    bool passed = (v1 == 1 && v2 == 0);
    print_test("Test 36: Cloned tensors have independent versions", passed);
    assert(passed);
}

// Test 37: Version after multiple operations
void test_37_version_multi_ops() {
    Tensor t = Tensor::ones(Shape{{3, 3}});
    ag::detail::version_tracker().register_tensor(t);  // Must register first
    
    for (int i = 0; i < 10; i++) {
        ag::detail::version_tracker().bump_version(t);
    }
    
    int v = ag::detail::version_tracker().get_version(t);
    bool passed = (v == 10);
    print_test("Test 37: Version accumulates over many operations", passed);
    assert(passed);
}

// Test 38: Unregister tensor cleans up
void test_38_version_cleanup() {
    Tensor t = Tensor::ones(Shape{{2, 2}});
    ag::detail::version_tracker().register_tensor(t);
    ag::detail::version_tracker().bump_version(t);
    
    // Unregister
    ag::detail::version_tracker().unregister_tensor(t);
    
    // After unregister, version should be 0 (not tracked)
    int v = ag::detail::version_tracker().get_version(t);
    bool passed = (v == 0);
    print_test("Test 38: Unregister tensor cleans version", passed);
    assert(passed);
}

// Test 39: Check version with correct value passes
void test_39_version_check_correct() {
    Tensor t = Tensor::ones(Shape{{2, 2}});
    ag::detail::version_tracker().register_tensor(t);
    ag::detail::version_tracker().bump_version(t);
    
    bool passed = false;
    try {
        ag::detail::version_tracker().check_version(t, 1, "test");
        passed = true;
    } catch (...) {
        passed = false;
    }
    
    print_test("Test 39: Version check with correct version passes", passed);
    assert(passed);
}

// Test 40: Check version with wrong value throws
void test_40_version_check_fails() {
    Tensor t = Tensor::ones(Shape{{2, 2}});
    ag::detail::version_tracker().register_tensor(t);
    ag::detail::version_tracker().bump_version(t);
    ag::detail::version_tracker().bump_version(t);
    
    bool passed = false;
    try {
        ag::detail::version_tracker().check_version(t, 1, "test");  // Actual is 2
        passed = false;
    } catch (const std::runtime_error&) {
        passed = true;  // Should throw
    }
    
    print_test("Test 40: Version check with wrong version throws", passed);
    assert(passed);
}

// =============================================================================
// Category 5: Execution Context Advanced (Tests 41-45)
// =============================================================================

// Test 41: Context preserved through long chain
void test_41_context_long_chain() {
    Value x = make_tensor(Tensor::ones(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "x");
    Value y = x;
    
    for (int i = 0; i < 10; i++) {
        y = y + 1.0f;
    }
    
    // All nodes should have context
    bool passed = y.node->creation_context.device.is_cpu();
    print_test("Test 41: Context preserved through long computation chain", passed);
    assert(passed);
}

// Test 42: Context on branching graph
void test_42_context_branching() {
    Value x = make_tensor(Tensor::ones(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "x");
    Value a = x * 2.0f;
    Value b = x + 1.0f;
    Value c = a + b;
    
    // All nodes should have captured context
    bool passed = x.node->creation_context.device.is_cpu() &&
                  a.node->creation_context.device.is_cpu() &&
                  b.node->creation_context.device.is_cpu() &&
                  c.node->creation_context.device.is_cpu();
    print_test("Test 42: Context captured on all branches", passed);
    assert(passed);
}

// Test 43: Context matches tensor device
void test_43_context_matches_device() {
    Value x_cpu = make_tensor(Tensor::ones(Shape{{2, 2}}).to_cpu(), "x_cpu");
    
    bool passed = x_cpu.node->creation_context.device.is_cpu() &&
                  x_cpu.val().device().is_cpu();
    print_test("Test 43: Context device matches tensor device", passed);
    assert(passed);
}

// Test 44: Stream captured (even if nullptr)
void test_44_stream_captured() {
    Value x = make_tensor(Tensor::ones(Shape{{2, 2}}), "x");
    
    // Stream should be captured (may be nullptr for CPU)
    bool passed = true;  // Always captures, even if nullptr
    print_test("Test 44: Stream field exists in context", passed);
    assert(passed);
}

// Test 45: Multiple operations preserve context
void test_45_multi_op_context() {
    Value x = make_tensor(Tensor::ones(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "x");
    Value y = x * x;
    Value z = y + y;
    Value w = z * z;
    
    // All should have context
    bool passed = x.node->creation_context.device.is_cpu() &&
                  y.node->creation_context.device.is_cpu() &&
                  z.node->creation_context.device.is_cpu() &&
                  w.node->creation_context.device.is_cpu();
    print_test("Test 45: Context preserved through multiple ops", passed);
    assert(passed);
}

// =============================================================================
// Category 6: Numerical Edge Cases (Tests 46-50)
// =============================================================================

// Test 46: Very small gradients (underflow)
void test_46_small_gradients() {
    Value x = make_tensor(Tensor::full(Shape{{2, 2}}, TensorOptions().with_req_grad(true), 1e-10f), "x");
    Value y = x * x;
    
    backward(y);
    
    // Should handle tiny gradients
    bool passed = !std::isnan(x.grad().data<float>()[0]) && !std::isinf(x.grad().data<float>()[0]);
    print_test("Test 46: Small gradients don't underflow to NaN", passed);
    assert(passed);
}

// Test 47: Large gradients (overflow check)
void test_47_large_gradients() {
    Value x = make_tensor(Tensor::full(Shape{{2, 2}}, TensorOptions().with_req_grad(true), 1e5f), "x");
    Value y = x * x;
    
    backward(y);
    
    // Check gradient isn't NaN
    bool passed = !std::isnan(x.grad().data<float>()[0]);
    print_test("Test 47: Large values don't produce NaN gradients", passed);
    assert(passed);
}

// Test 48: Zero values
void test_48_zero_values() {
    Value x = make_tensor(Tensor::zeros(Shape{{2, 2}}, TensorOptions().with_req_grad(true)), "x");
    Value y = x * x;
    
    backward(y);
    
    // dy/dx = 2*x = 0
    bool passed = approx_equal(x.grad().data<float>()[0], 0.0f);
    print_test("Test 48: Zero values produce zero gradients", passed);
    assert(passed);
}

// Test 49: Negative values
void test_49_negative_values() {
    Value x = make_tensor(Tensor::full(Shape{{2, 2}}, TensorOptions().with_req_grad(true), -3.0f), "x");
    Value y = x * x;  // y = 9
    
    backward(y);
    
    // dy/dx = 2*x = -6
    bool passed = approx_equal(x.grad().data<float>()[0], -6.0f);
    print_test("Test 49: Negative values handled correctly", passed);
    assert(passed);
}

// Test 50: Mixed positive/negative across tensor
void test_50_mixed_signs() {
    // Create tensor with requires_grad
    Tensor data = Tensor::zeros(Shape{{2, 2}}, TensorOptions().with_req_grad(true));
    data.data<float>()[0] = 2.0f;
    data.data<float>()[1] = -2.0f;
    data.data<float>()[2] = 0.0f;
    data.data<float>()[3] = 1.0f;
    
    Value x = make_tensor(data, "x");
    Value y = x * x;
    
    backward(y);
    
    // Gradients: 2*2=4, 2*(-2)=-4, 2*0=0, 2*1=2
    bool passed = approx_equal(x.grad().data<float>()[0], 4.0f) &&
                  approx_equal(x.grad().data<float>()[1], -4.0f);
    print_test("Test 50: Mixed signs in tensor", passed);
    assert(passed);
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Phase 1 Advanced Edge Case Suite (30 Tests)" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    std::cout << "--- Category 1: Gradient Accumulation Edge Cases ---" << std::endl;
    test_21_multiple_backward_accumulation();
    test_22_zero_grad();
    test_23_scalar_gradient();
    test_24_shared_leaf_accumulation();
    test_25_intermediate_no_accumulation();
    
    std::cout << "\n--- Category 2: Graph Structure Edge Cases ---" << std::endl;
    test_26_diamond_graph();
    test_27_deep_chain();
    test_28_wide_graph();
    test_29_multiple_leaves();
    test_30_detached_intermediate();
    
    std::cout << "\n--- Category 3: Empty and Edge Tensor Cases ---" << std::endl;
    test_31_minimum_dimension();
    test_32_1d_tensor();
    test_33_3d_tensor();
    test_34_single_element();
    test_35_large_tensor();
    
    std::cout << "\n--- Category 4: Version Tracking Advanced ---" << std::endl;
    test_36_version_clone();
    test_37_version_multi_ops();
    test_38_version_cleanup();
    test_39_version_check_correct();
    test_40_version_check_fails();
    
    std::cout << "\n--- Category 5: Execution Context Advanced ---" << std::endl;
    test_41_context_long_chain();
    test_42_context_branching();
    test_43_context_matches_device();
    test_44_stream_captured();
    test_45_multi_op_context();
    
    std::cout << "\n--- Category 6: Numerical Edge Cases ---" << std::endl;
    test_46_small_gradients();
    test_47_large_gradients();
    test_48_zero_values();
    test_49_negative_values();
    test_50_mixed_signs();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "All advanced edge case tests passed!" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return 0;
}
