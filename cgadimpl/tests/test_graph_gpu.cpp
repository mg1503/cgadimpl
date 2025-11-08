#include "ad/ag_all.hpp"
#include <iostream>

using namespace ag;
using namespace OwnTensor;

// Simple model for CUDA graph capture
class SimpleMLP : public nn::Module {
public:
    nn::Linear fc1, fc2;

    // --- FIX #1: Constructor must take Device ---
    SimpleMLP(int in, int hid, int out, Device dev)
        : fc1(in, hid, dev), fc2(hid, out, dev) {
        for(auto& p : fc1.parameters()) params_.push_back(p);
        for(auto& p : fc2.parameters()) params_.push_back(p);
    }

    // --- FIX #2: Un-hide the base class's operator() overloads ---
    using nn::Module::operator();

    // --- FIX #3: Signature must match the base class pure virtual function ---
    Value operator()(Value x) override {
        x = fc1(x);
        x = ag::relu(x);
        x = fc2(x);
        return x;
    }
};

// --- FIX #4: Function signature must use Device ---

void test_cuda_graph(Device device) {
    if (device != Device::CUDA) return;
    std::cout << "\n--- Testing CUDA Graph Capture ---\n";

    // --- FIX: Create the model BEFORE capturing ---
    SimpleMLP model(10, 20, 5, device);
    Tensor x_tensor = Tensor::randn(Shape{{4, 10}}, TensorOptions().with_device(device));
    // --- END FIX ---

    CudaGraphRunner runner;

    // Capture
    std::cout << "Beginning capture...\n";
    runner.begin_capture();
    Value y1 = model(x_tensor); // Only the forward pass is inside the capture block
    (void)y1;
    runner.end_capture();
    std::cout << "Capture successful.\n";

    // Replay
    std::cout << "Replaying graph...\n";
    bool ok = runner.replay();
    std::cout << "Replay " << (ok ? "successful" : "failed") << ".\n";
}
int main() {
    if (device::cuda_available()) {
        // --- FIX #5: Call the test with the correct Device enum value ---
        test_cuda_graph(Device::CUDA);
    } else {
        std::cout << "CUDA not available, skipping CUDA graph test.\n";
    }

    std::cout << "\n✅ CUDA graph test completed.\n";
    return 0;
}