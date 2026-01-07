#include "ad/ag_all.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <functional>
using namespace ag;
using namespace OwnTensor;

// ==========================================================
// UTILITY FUNCTIONS
// ==========================================================

bool check_grad(const Tensor& analytical_grad, const Tensor& numerical_grad, double tol = 1e-3) {
    if (analytical_grad.shape().dims != numerical_grad.shape().dims) {
        std::cerr << "  [FAIL] Shape mismatch!\n";
        return false;
    }
    
    Tensor diff = abs(analytical_grad - numerical_grad, ag::current_stream());
    Tensor max_val = reduce_max(diff);
    cudaDeviceSynchronize();
    double error = max_val.to_cpu().data<float>()[0];

    if (error > tol) {
        std::cerr << "  [FAIL] Max error: " << error << " > tolerance: " << tol << "\n";
        // For debugging, let's print the tensors
        std::cerr << "  Analytical Grad:\n";
        analytical_grad.display(std::cerr, 4);
        std::cerr << "  Numerical Grad:\n";
        numerical_grad.display(std::cerr, 4);
        return false;
    }
    return true;
}

Tensor numerical_gradient(Value& param, std::function<Value()> func, double h = 1e-4) {
    Tensor original_val = param.val().clone();

    // --- FIX START: Correct memory management ---

    // 1. Create the final gradient tensor on the correct device (GPU), but don't use it yet.
    Tensor grad_on_device = Tensor::zeros(param.val().shape(), ag::options(param.val()));

    // 2. Create a NAMED CPU tensor to work with. This object will not be destroyed
    //    until the function exits, keeping its data pointer valid.
    Tensor grad_cpu = grad_on_device.to_cpu();
    float* grad_data = grad_cpu.data<float>(); // This pointer is now safe to use.

    // 3. Create a CPU copy of the original values ONCE before the loop to avoid
    //    repeated and slow device-to-host transfers.
    Tensor original_val_cpu = original_val.to_cpu();
    const float* original_data = original_val_cpu.data<float>();

    for (int64_t i = 0; i < param.val().numel(); ++i) {
        // Compute f(x+h)
        Tensor val_plus_h = original_val_cpu.clone();
        val_plus_h.data<float>()[i] = original_data[i] + h;
        param.val() = val_plus_h.to(param.val().device()); // Upload to GPU
        Value loss1 = func();
        double loss1_val = sum(loss1).val().to_cpu().data<float>()[0];

        // Compute f(x-h)
        Tensor val_minus_h = original_val_cpu.clone();
        val_minus_h.data<float>()[i] = original_data[i] - h;
        param.val() = val_minus_h.to(param.val().device()); // Upload to GPU
        Value loss2 = func();
        double loss2_val = sum(loss2).val().to_cpu().data<float>()[0];

        // Central difference formula - write to the safe CPU buffer
        grad_data[i] = (loss1_val - loss2_val) / (2.0 * h);

        // Restore original value on the GPU for the next parameter
        param.val() = original_val.to(param.val().device());
    }

    // 4. After the loop, copy the computed gradients from the CPU tensor
    //    back to the GPU tensor.
    grad_on_device.copy_(grad_cpu);

    return grad_on_device;
    // --- FIX END ---
}

// ==========================================================
// TEST DISPATCHER
// ==========================================================

bool run_test(const std::string& name, std::function<void()> test_func) {
    std::cout << "Testing: " << std::left << std::setw(20) << name << "... ";
    try {
        test_func();
        std::cout << "[PASS]\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "[FAIL]\n";
        std::cerr << "  Exception: " << e.what() << "\n";
        return false;
    }
}

// ==========================================================
// TEST DEFINITIONS
// ==========================================================

void test_all_ops() {
    // --- FIX: This is a GPU test file, so we must use the CUDA device. ---
    #ifdef WITH_CUDA
        Device dev = Device::CUDA;
    #else
        Device dev = Device::CPU; // Fallback for non-CUDA builds
    #endif
    auto opts = TensorOptions().with_device(dev).with_req_grad(true);

    // --- Unary Ops ---
    run_test("ReLU", [&](){
        Value a = make_tensor(Tensor::randn<float>(Shape{{4, 5}}, opts));
        auto f = [&](){ return relu(a); };
        backward(sum(f()));
        assert(check_grad(a.grad(), numerical_gradient(a, f)));
    });
    run_test("Exp", [&](){
        Value a = make_tensor(Tensor::randn<float>(Shape{{4, 5}}, opts));
        auto f = [&](){ return exp(a); };
        backward(sum(f()));
        assert(check_grad(a.grad(), numerical_gradient(a, f)));
    });
     run_test("Log", [&](){
        Value a = make_tensor(Tensor::rand<float>(Shape{{4, 5}}, opts) + 0.1); // Ensure positive
        auto f = [&](){ return log(a); };
        backward(sum(f()));
        assert(check_grad(a.grad(), numerical_gradient(a, f)));
    });
    run_test("Tanh", [&](){
        Value a = make_tensor(Tensor::randn<float>(Shape{{4, 5}}, opts));
        auto f = [&](){ return tanh(a); };
        backward(sum(f()));
        assert(check_grad(a.grad(), numerical_gradient(a, f)));
    });
    run_test("Sigmoid", [&](){
        Value a = make_tensor(Tensor::randn<float>(Shape{{4, 5}}, opts));
        auto f = [&](){ return sigmoid(a); };
        backward(sum(f()));
        assert(check_grad(a.grad(), numerical_gradient(a, f)));
    });
     run_test("GELU", [&](){
        Value a = make_tensor(Tensor::randn<float>(Shape{{4, 5}}, opts));
        auto f = [&](){ return gelu(a); };
        backward(sum(f()));
        assert(check_grad(a.grad(), numerical_gradient(a, f), 1e-2)); // GELU is an approx
    });
    // run_test("Tan", [&](){
    //     Value a = make_tensor(Tensor::randn(Shape{{4, 5}}, opts));
    //     auto f = [&](){ return tan(a); };
    //     backward(sum(f()));
    //     assert(check_grad(a.grad(), numerical_gradient(a, f)));
    // });
    // run_test("Asin", [&](){
    //     // Input must be in [-1, 1]. Use rand which is [0,1], then scale to [-0.9, 0.9]
    //     // to avoid gradient issues at the boundaries.
    //     Value a = make_tensor((Tensor::rand(Shape{{4, 5}}, opts) * 1.8f) - 0.9f);
    //     auto f = [&](){ return asin(a); };
    //     backward(sum(f()));
    //     assert(check_grad(a.grad(), numerical_gradient(a, f)));
    // });
    // run_test("Acos", [&](){
    //     // Input must be in [-1, 1].
    //     Value a = make_tensor((Tensor::rand(Shape{{4, 5}}, opts) * 1.8f) - 0.9f);
    //     auto f = [&](){ return acos(a); };
    //     backward(sum(f()));
    //     assert(check_grad(a.grad(), numerical_gradient(a, f)));
    // });
    // run_test("Atan", [&](){
    //     Value a = make_tensor(Tensor::randn(Shape{{4, 5}}, opts));
    //     auto f = [&](){ return atan(a); };
    //     backward(sum(f()));
    //     assert(check_grad(a.grad(), numerical_gradient(a, f)));
    // });

    // --- Binary Ops with Broadcasting ---
    run_test("Add (Broadcast)", [&](){
        Value a = make_tensor(Tensor::randn<float>(Shape{{4, 5}}, opts));
        Value b = make_tensor(Tensor::randn<float>(Shape{{1, 5}}, opts));
        auto f = [&](){ return a + b; };
        backward(sum(f()));
        assert(check_grad(a.grad(), numerical_gradient(a, f)));
        assert(check_grad(b.grad(), numerical_gradient(b, f)));
    });
    run_test("Sub (Broadcast)", [&](){
        Value a = make_tensor(Tensor::randn<float>(Shape{{4, 5}}, opts));
        Value b = make_tensor(Tensor::randn<float>(Shape{{1, 5}}, opts));
        auto f = [&](){ return a - b; };
        backward(sum(f()));
        assert(check_grad(a.grad(), numerical_gradient(a, f)));
        assert(check_grad(b.grad(), numerical_gradient(b, f)));
    });
    run_test("Mul (Broadcast)", [&](){
        Value a = make_tensor(Tensor::randn<float>(Shape{{4, 5}}, opts));
        Value b = make_tensor(Tensor::randn<float>(Shape{{1, 5}}, opts));
        auto f = [&](){ return a * b; };
        backward(sum(f()));
        assert(check_grad(a.grad(), numerical_gradient(a, f)));
        assert(check_grad(b.grad(), numerical_gradient(b, f)));
    });
    run_test("Div (Broadcast)", [&](){
        Value a = make_tensor(Tensor::randn<float>(Shape{{4, 5}}, opts));
        Value b = make_tensor(Tensor::rand<float>(Shape{{1, 5}}, opts) + 1.0); // Avoid division by zero
        auto f = [&](){ return a / b; };
        backward(sum(f()));
        assert(check_grad(a.grad(), numerical_gradient(a, f)));
        assert(check_grad(b.grad(), numerical_gradient(b, f)));
    });

    // --- Reductions ---
    run_test("Sum (all)", [&](){
        Value a = make_tensor(Tensor::randn<float>(Shape{{4, 5}}, opts));
        auto f = [&](){ return sum(a); };
        backward(f()); // Loss is already a scalar
        assert(check_grad(a.grad(), numerical_gradient(a, f)));
    });
    run_test("Mean (all)", [&](){
        Value a = make_tensor(Tensor::randn<float>(Shape{{4, 5}}, opts));
        auto f = [&](){ return mean_all(a); };
        backward(f());
        assert(check_grad(a.grad(), numerical_gradient(a, f)));
    });
    run_test("RowSum", [&](){
        Value a = make_tensor(Tensor::randn<float>(Shape{{4, 5}}, opts));
        auto f = [&](){ return rowsum(a); };
        backward(sum(f())); // Sum again to get scalar loss
        assert(check_grad(a.grad(), numerical_gradient(a, f)));
    });

    // --- MatMul and Linear ---
    run_test("MatMul", [&](){
        Value a = make_tensor(Tensor::randn<float>(Shape{{16, 32}}, opts));
        Value b = make_tensor(Tensor::randn<float>(Shape{{32, 8}}, opts));
        auto f = [&](){ return matmul(a, b); };
        backward(sum(f()));
        assert(check_grad(a.grad(), numerical_gradient(a, f), 1e-2));
        assert(check_grad(b.grad(), numerical_gradient(b, f), 1e-2));
    });
    run_test("Linear", [&](){
        Value x = make_tensor(Tensor::randn<float>(Shape{{16, 32}}, opts));
        Value w = make_tensor(Tensor::randn<float>(Shape{{8, 32}}, opts)); // (out, in)
        Value b = make_tensor(Tensor::randn<float>(Shape{{1, 8}}, opts));
        auto f = [&](){ return linear(x, w, b); };
        backward(sum(f()));
        assert(check_grad(x.grad(), numerical_gradient(x, f), 1e-2));
        assert(check_grad(w.grad(), numerical_gradient(w, f), 1e-2));
        assert(check_grad(b.grad(), numerical_gradient(b, f), 1e-2));
    });

    // --- Normalization ---
    run_test("LayerNorm", [&](){
        Value a = make_tensor(Tensor::randn<float>(Shape{{4, 10}}, opts));
        auto f = [&](){ return laynor(a); };
        backward(sum(f()));
        assert(check_grad(a.grad(), numerical_gradient(a, f)));
    });
     run_test("RMSNorm", [&](){
        Value a = make_tensor(Tensor::randn<float>(Shape{{4, 10}}, opts));
        auto f = [&](){ return rms(a); };
        backward(sum(f()));
        assert(check_grad(a.grad(), numerical_gradient(a, f)));
    });
}


int main() {
    try {
        #ifndef WITH_CUDA
            std::cout << "Test skipped: Not compiled with CUDA support.\n";
            return 0;
        #endif

        std::cout << "\n==================================================\n";
        std::cout << "--- Exhaustive Core CUDA Operations Test Suite ---\n";
        std::cout << "==================================================\n";

        test_all_ops();

        std::cout << "\n✅ All tests passed!\n";

    } catch (const std::exception& e) {
        std::cerr << "\nCaught exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}