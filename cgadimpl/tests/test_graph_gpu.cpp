// =========================================================
// FILE: cgadimpl/tests/test_mlp_training.cpp
// A simple, standalone test for the full framework on the CPU.
// =========================================================
#include "ad/ag_all.hpp"


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


int main() {
    // --- SETUP (same as your GPU test) ---
    using namespace ag;
    const Device device = Device::CUDA;
    SimpleMLP model;
    model.to(device);
    Value input = make_tensor(Tensor::randn(8, 10, 1337, device), "input");
    Value labels = make_tensor(Tensor::zeros(8, 5, device), "labels");
    std::cout << "Data created successfully." << std::endl;

    CudaGraphRunner graph_runner;

    // --- WARMUP (Crucial!) ---
    // Run the training step once or twice eagerly to let cuBLAS
    // and other libraries initialize and allocate any internal memory.
    std::cout << "Warming up..." << std::endl;
    for(int i = 0; i < 2; ++i) {
        model.zero_grad();
        Value output = model(input);
        Value loss = mse_loss(output, labels);
        backward(loss);
        // optimizer.step();
    }
    cudaDeviceSynchronize(); // Wait for warmup to finish

    // --- CAPTURE ---
    std::cout << "Capturing graph..." << std::endl;
    graph_runner.begin_capture();
        // This code block is NOT executed by the CPU. It is recorded by CUDA.
        model.zero_grad();
        Value output_captured = model(input);
        Value loss_captured = mse_loss(output_captured, labels);
        backward(loss_captured);
        // optimizer.step();
    graph_runner.end_capture();
    std::cout << "Graph captured successfully." << std::endl;

    // --- HIGH-PERFORMANCE REPLAY LOOP ---
    std::cout << "Starting replay loop..." << std::endl;
    for(int i = 0; i < 1000; ++i) {
        graph_runner.replay(); // Single, low-overhead launch
    }
    cudaDeviceSynchronize(); // Wait for all replays to finish

    std::cout << "\nSUCCESS: CUDA Graph captured and replayed." << std::endl;
    return 0;
}