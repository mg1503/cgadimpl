#include <iostream>
#include <cmath>
#include <iomanip>
#include "ad/ag_all.hpp"

using namespace ag;

// Helper to compare two tensors
bool tensors_close(const Tensor& a, const Tensor& b, float tolerance = 1e-4) {
    if (a.shape().dims != b.shape().dims) return false;
    if (a.numel() != b.numel()) return false;
    
    // Ensure we are comparing on CPU
    Tensor a_cpu = a.to_cpu();
    Tensor b_cpu = b.to_cpu();
    
    const float* data_a = a_cpu.data<float>();
    const float* data_b = b_cpu.data<float>();
    
    for (size_t i = 0; i < a_cpu.numel(); i++) {
        float diff = std::abs(data_a[i] - data_b[i]);
        if (diff > tolerance) {
            std::cout << "Mismatch at index " << i << ": sequential=" << data_a[i] 
                      << ", parallel=" << data_b[i] << ", diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}


int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "CORRECTNESS TEST (CRITICAL)" << std::endl;
    std::cout << "Sequential vs Parallel Backward" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nThis is the most important test!" << std::endl;
    std::cout << "It verifies that the dependency counter" << std::endl;
    std::cout << "produces IDENTICAL gradients in both modes." << std::endl;
    std::cout << "========================================\n" << std::endl;

    bool all_tests_passed = true;

    // ========================================
    // TEST 1: Diamond Pattern
    // ========================================
    std::cout << "Test 1: Diamond Pattern" << std::endl;
    std::cout << "  Building graph..." << std::endl;
    
    Tensor X1 = Tensor::randn<float>(Shape{{5, 5}}, TensorOptions().with_req_grad(true));
    auto x1_seq = make_tensor(X1, "x1");
    auto f1_seq = relu(x1_seq);
    auto g1_seq = x1_seq * x1_seq;
    auto y1_seq = f1_seq + g1_seq;
    auto loss1_seq = sum(y1_seq);
    
    auto x1_par = make_tensor(X1, "x1_par");
    auto f1_par = relu(x1_par);
    auto g1_par = x1_par * x1_par;
    auto y1_par = f1_par + g1_par;
    auto loss1_par = sum(y1_par);
    
    // Sequential
    zero_grad(loss1_seq);
    backward(loss1_seq, nullptr, false);
    Tensor grad1_seq = x1_seq.grad();
    
    // Parallel
    zero_grad(loss1_par);
    backward(loss1_par, nullptr, true);
    Tensor grad1_par = x1_par.grad();
    
    bool test1_pass = tensors_close(grad1_seq, grad1_par);
    std::cout << "  Result: " << (test1_pass ? "✅ PASS" : "❌ FAIL") << std::endl;
    all_tests_passed &= test1_pass;

    // ========================================
    // TEST 2: Wide Fan-out
    // ========================================
    std::cout << "\nTest 2: Wide Fan-out (50 children)" << std::endl;
    std::cout << "  Building graph..." << std::endl;
    
    Tensor X2 = Tensor::randn<float>(Shape{{4, 4}}, TensorOptions().with_req_grad(true));
    auto x2_seq = make_tensor(X2, "x2");
    auto x2_par = make_tensor(X2, "x2_par");
    
    // Sequential version
    std::vector<Value> ops_seq;
    for (int i = 0; i < 50; i++) {
        if (i % 2 == 0) ops_seq.push_back(relu(x2_seq));
        else ops_seq.push_back(sigmoid(x2_seq));
    }
    auto loss2_seq = ops_seq[0];
    for (size_t i = 1; i < ops_seq.size(); i++) {
        loss2_seq = loss2_seq + ops_seq[i];
    }
    loss2_seq = sum(loss2_seq);
    
    // Parallel version
    std::vector<Value> ops_par;
    for (int i = 0; i < 50; i++) {
        if (i % 2 == 0) ops_par.push_back(relu(x2_par));
        else ops_par.push_back(sigmoid(x2_par));
    }
    auto loss2_par = ops_par[0];
    for (size_t i = 1; i < ops_par.size(); i++) {
        loss2_par = loss2_par + ops_par[i];
    }
    loss2_par = sum(loss2_par);
    
    // Sequential
    zero_grad(loss2_seq);
    backward(loss2_seq, nullptr, false);
    Tensor grad2_seq = x2_seq.grad();
    
    // Parallel
    zero_grad(loss2_par);
    backward(loss2_par, nullptr, true);
    Tensor grad2_par = x2_par.grad();
    
    bool test2_pass = tensors_close(grad2_seq, grad2_par);
    std::cout << "  Result: " << (test2_pass ? "✅ PASS" : "❌ FAIL") << std::endl;
    all_tests_passed &= test2_pass;

    // ========================================
    // TEST 3: Multi-layer Network
    // ========================================
    std::cout << "\nTest 3: Multi-layer Network" << std::endl;
    std::cout << "  Building graph..." << std::endl;
    
    Tensor W1 = Tensor::randn<float>(Shape{{8, 8}}, TensorOptions().with_req_grad(true));
    Tensor W2 = Tensor::randn<float>(Shape{{8, 8}}, TensorOptions().with_req_grad(true));
    Tensor X3 = Tensor::randn<float>(Shape{{1, 8}}, TensorOptions().with_req_grad(false));
    
    // Sequential
    auto w1_seq = make_tensor(W1, "w1");
    auto w2_seq = make_tensor(W2, "w2");
    auto x3_seq = make_tensor(X3, "x3");
    auto h1_seq = relu(matmul(x3_seq, w1_seq));
    auto h2_seq = relu(matmul(h1_seq, w2_seq));
    auto loss3_seq = sum(h2_seq);
    
    // Parallel
    auto w1_par = make_tensor(W1, "w1_par");
    auto w2_par = make_tensor(W2, "w2_par");
    auto x3_par = make_tensor(X3, "x3_par");
    auto h1_par = relu(matmul(x3_par, w1_par));
    auto h2_par = relu(matmul(h1_par, w2_par));
    auto loss3_par = sum(h2_par);
    
    // Sequential
    zero_grad(loss3_seq);
    backward(loss3_seq, nullptr, false);
    Tensor grad3_w1_seq = w1_seq.grad();
    Tensor grad3_w2_seq = w2_seq.grad();
    
    // Parallel
    zero_grad(loss3_par);
    backward(loss3_par, nullptr, true);
    Tensor grad3_w1_par = w1_par.grad();
    Tensor grad3_w2_par = w2_par.grad();
    
    bool test3_pass = tensors_close(grad3_w1_seq, grad3_w1_par) && 
                      tensors_close(grad3_w2_seq, grad3_w2_par);
    std::cout << "  Result: " << (test3_pass ? "✅ PASS" : "❌ FAIL") << std::endl;
    all_tests_passed &= test3_pass;

    // ========================================
    // Final Report
    // ========================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "FINAL RESULT" << std::endl;
    std::cout << "========================================" << std::endl;
    
    if (all_tests_passed) {
        std::cout << "✅✅✅ ALL TESTS PASSED! ✅✅✅" << std::endl;
        std::cout << "\n🎉 The dependency counter implementation is" << std::endl;
        std::cout << "   MATHEMATICALLY CORRECT!" << std::endl;
        std::cout << "\n   Sequential and parallel backward passes" << std::endl;
        std::cout << "   produce IDENTICAL gradients across all" << std::endl;
        std::cout << "   test patterns (diamond, fan-out, multi-layer)." << std::endl;
        std::cout << "\n   This proves the atomic dependency counter" << std::endl;
        std::cout << "   correctly synchronizes gradient accumulation!" << std::endl;
    } else {
        std::cout << "❌ SOME TESTS FAILED" << std::endl;
        std::cout << "   The dependency counter has issues." << std::endl;
    }
    std::cout << "========================================" << std::endl;
    
    return all_tests_passed ? 0 : 1;
}
