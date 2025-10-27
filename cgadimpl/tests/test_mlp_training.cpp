// =========================================================
// FILE: cgadimpl/tests/test_mlp_training.cpp
// A simple, standalone test for the full framework on the CPU.
// =========================================================

#include "nn/nn.hpp"
#include "ad/autodiff.hpp"
#include "ad/debug.hpp" // For printing
#include <iostream>
#include <cassert>

// --- A Simple MLP Model ---
class SimpleMLP : public ag::nn::Module {
public:
    ag::nn::Linear fc1;
    ag::nn::ReLU relu1;
    ag::nn::Linear fc2;
    ag::nn::ReLU relu2;
    ag::nn::Linear fc3;

    SimpleMLP() : fc1(10, 20), fc2(20, 20), fc3(20, 5) {
        // Manually collect parameters from sub-modules for now
        params_.insert(params_.end(), fc1.parameters().begin(), fc1.parameters().end());
        params_.insert(params_.end(), fc2.parameters().begin(), fc2.parameters().end());
        params_.insert(params_.end(), fc3.parameters().begin(), fc3.parameters().end());
    }

    // The forward pass that fulfills the Module contract
    ag::Value operator()(const ag::Value& x) override {
        auto h = relu1(fc1(x));
        h = relu2(fc2(h));
        return fc3(h);
    }
};


// --- The Main Test Function ---
int main() {
    using namespace ag;
    std::cout << "==========================================================" << std::endl;
    std::cout << "--- Starting End-to-End CPU Training Test ---" << std::endl;
    std::cout << "==========================================================" << std::endl;
    // 1. Build the model. It is created on the CPU by default.
    SimpleMLP model;
    std::cout << "Model created successfully." << std::endl;

    // 2. Create CPU data
    Value input = make_tensor(Tensor::randn(8, 10, 1337,Device::CPU), "input"); // Batch size 8
    Value labels = make_tensor(Tensor::zeros(8, 5, Device::CPU), "labels");
    std::cout << "Data created successfully." << std::endl;

    // 3. Run a full training step
    std::cout << "Performing forward pass..." << std::endl;
    model.zero_grad();
    Value output = model(input);

    std::cout << "Calculating loss..." << std::endl;
    Value loss = mse_loss(output, labels);

    std::cout << "Performing backward pass..." << std::endl;
    backward(loss);
    std::cout << "Backward pass complete." << std::endl;

    // 4. Verify that gradients were computed
    const auto& params = model.parameters();
    assert(params.size() == 4); // W1, b1, W2, b2
    
    // Check the gradient of the first weight matrix
    const auto& W1_grad = params[0].grad();
    float grad_sum = W1_grad.sum_scalar();

    std::cout << "Sum of gradients for the first weight matrix: " << grad_sum << std::endl;
    
    // A simple but effective check: if backprop works, the gradients won't be zero.
    assert(grad_sum != 0.0f);
    
    std::cout << "\nSUCCESS: The framework correctly performed an end-to-end training step on the CPU." << std::endl;

    std::cout << "\n--- Now testing model and data on GPU (if available) ---" << std::endl;

    const Device device = Device::CUDA; // Use a variable to make it easy

    SimpleMLP model2;    
    model.to(device); // Move the model to the GPU
    std::cout << "Model created and moved to GPU successfully." << std::endl;

    // Create data directly on the GPU
    Value input2 = make_tensor(Tensor::randn(8, 10, 1337, device), "input");
    Value labels2 = make_tensor(Tensor::zeros(8, 5, device), "labels");
    std::cout << "Data created successfully." << std::endl;
    std::cout << "Performing forward pass on GPU..." << std::endl;
    model2.zero_grad();
    Value output2 = model2(input2); 
    std::cout << "Calculating loss on GPU..." << std::endl;
    Value loss2 = mse_loss(output2, labels2);
    std::cout << "Performing backward pass on GPU..." << std::endl;

    backward(loss2);
    std::cout << "Backward pass complete." << std::endl;    

    // Verify gradients on GPU
    const auto& params2 = model2.parameters();
    assert(params2.size() == 4); // W1, b1, W2, b2
    const auto& W1_grad2 = params2[0].grad();
    float grad_sum2 = W1_grad2.sum_scalar();

    std::cout << "Sum of gradients for the first weight matrix on GPU: " << grad_sum2 << std::endl;
    assert(grad_sum2 != 0.0f);
    std::cout << "\nSUCCESS: The framework correctly performed an end-to-end training step on the GPU." << std::endl;
    


    return 0; // Return 0 on success
}