#include "ad/ag_all.hpp"
#include <iostream>
#include <chrono>

using namespace ag;

struct ManualMLP : nn::Module {
    ManualMLP(int64_t in_dim, int64_t hid_dim, int64_t out_dim) {
        auto param_opts = OwnTensor::TensorOptions().with_req_grad(true);
        
        // Initialize weights and biases
        w1 = make_tensor(OwnTensor::Tensor::randn(Shape{{in_dim, hid_dim}}, param_opts), "w1");
        b1 = make_tensor(OwnTensor::Tensor::randn(Shape{{1, hid_dim}}, param_opts), "b1");
        w2 = make_tensor(OwnTensor::Tensor::randn(Shape{{hid_dim, out_dim}}, param_opts), "w2");
        b2 = make_tensor(OwnTensor::Tensor::randn(Shape{{1, out_dim}}, param_opts), "b2");

        params_ = {w1, b1, w2, b2};
    }

    Value operator()(Value x) override {
        // Layer 1: matmul + add + relu
        x = add(matmul(x, w1), b1);
        x = relu(x);
        
        // Layer 2: matmul + add
        x = add(matmul(x, w2), b2);
        return x;
    }

    Value w1, b1, w2, b2;
};

int main() {
    Device device = Device::CUDA;
    auto model = std::make_shared<ManualMLP>(784, 1024, 10);
    model->to(device);

    auto input_opts = OwnTensor::TensorOptions().with_device(device);
    Value input = make_tensor(OwnTensor::Tensor::randn(Shape{{32, 784}}, input_opts), "input");

    // Warm-up pass
    {
        Value output = (*model)(input);
        cudaDeviceSynchronize();
    }

    std::cout << "Starting 5 sequential forward passes..." << std::endl;

    for (int i = 1; i <= 5; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        Value output = (*model)(input);
        cudaDeviceSynchronize(); 

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Pass " << i << " Time: " << duration.count() << " ms" << std::endl;
    }

    return 0;
}