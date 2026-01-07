// test_end_to_end_gpu.cpp
#include "ad/ag_all.hpp"
#include <iostream>

using namespace ag;
using namespace OwnTensor;

// A simple MLP for testing
class SimpleMLP : public nn::Module {
public:
    nn::Linear fc1, fc2;
    // --- FIX: Constructor now takes ag::Device ---
    SimpleMLP(int in, int hid, int out, Device dev)
        : fc1(in, hid, dev), fc2(hid, out, dev) {
        // This parameter collection logic can be improved, but is functional for now
        for(auto& p : fc1.parameters()) params_.push_back(p);
        for(auto& p : fc2.parameters()) params_.push_back(p);
    }
    using nn::Module::operator();
    
    // This signature is correct (matches the base class)
    Value operator()(Value x) override {
        return fc2(ag::relu(fc1(x)));
    }
};

// --- FIX: Function now takes ag::Device ---
void test_e2e_on_device(Device dev) {
    std::cout << "\n--- Testing End-to-End on " << (dev == Device::CUDA ? "CUDA" : "CPU") << " ---\n";
    
    // This now passes the correct device type
    SimpleMLP model(10, 5, 2, dev);

    // --- FIX: Use the convenient Module::operator()(const Tensor&) overload ---
    // This automatically handles the conversion from Tensor to Value.
    Tensor x_tensor = Tensor::randn<float>(Shape{{4, 10}}, TensorOptions().with_device(dev));
    Value y = model(x_tensor);
    // --- END FIX ---

    std::cout << "Forward pass successful on " << (dev == Device::CUDA ? "CUDA" : "CPU") << ".\n";
    
    // Dummy loss and backward to test VJP path
    Value loss = sum(y);
    backward(loss);
    std::cout << "Backward pass successful.\n";
}

int main() {
    test_e2e_on_device(Device::CPU);
    
    // --- FIX: Use ag::Device for the check ---
    if (device::cuda_available()) {
        test_e2e_on_device(Device::CUDA);
    } else {
        std::cout << "CUDA not available, skipping GPU test.\n";
    }
    
    std::cout << "\n✅ End-to-end test completed successfully.\n";
    return 0;
}