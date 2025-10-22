#include "nn/nn.hpp"
#include "ad/ag_all.hpp"
#include <iostream>

int main() {
    using namespace ag;

    // 1. Create a linear layer on the CPU
    nn::Linear fc1(128, 64, Device::CPU);

    // 2. Move the entire layer (weights and biases) to the GPU
    std::cout << "Moving model to GPU..." << std::endl;
    fc1.to(Device::CUDA);

    // 3. Create some input data directly on the GPU
    Tensor x_tensor = Tensor::randn(10, 128, /*seed=*/1337, Device::CUDA);
    Value x = constant(x_tensor, "input");

    // 4. Perform a forward pass on the GPU
    std::cout << "Performing forward pass on GPU..." << std::endl;
    Value y = fc1(x);

    // The result 'y' will have its value tensor on the GPU
    std::cout << "Output is on CUDA: " << (y.val().is_cuda() ? "true" : "false") << std::endl;
    std::cout << "Output shape: " << y.shape().first << "x" << y.shape().second << std::endl;
    
    // You can now proceed with loss calculation and backpropagation
    // e.g., backward(y);

    return 0;
}