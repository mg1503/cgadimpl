#include <torch/torch.h>
#include <iostream>
#include <chrono>

struct ManualMLP : torch::nn::Module {
    ManualMLP(int64_t in_dim, int64_t hid_dim, int64_t out_dim) {
        w1 = register_parameter("w1", torch::randn({in_dim, hid_dim}));
        b1 = register_parameter("b1", torch::randn({hid_dim}));
        w2 = register_parameter("w2", torch::randn({hid_dim, out_dim}));
        b2 = register_parameter("b2", torch::randn({out_dim}));
    }

    // DAG using only matmul and add
    torch::Tensor forward(torch::Tensor x) {
        // First Layer
        x = torch::add(torch::matmul(x, w1), b1);
        x = torch::relu(x);
        // Second Layer
        x = torch::add(torch::matmul(x, w2), b2);
        return x;
    }

    torch::Tensor w1, b1, w2, b2;
};

int main() {
    torch::Device device(torch::kCUDA);
    auto model = std::make_shared<ManualMLP>(784, 1024, 10);
    model->to(device);

    torch::Tensor input = torch::randn({32, 784}).to(device);

    // Warm-up pass (ensures kernels are loaded before timing)
    {
        at::NoGradGuard no_grad;
        model->forward(input);
        torch::cuda::synchronize();
    }

    std::cout << "Starting 5 sequential forward passes..." << std::endl;

    for (int i = 1; i <= 5; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        // Sequential Forward Pass
        torch::Tensor output = model->forward(input);

        // Synchronize is MANDATORY to get actual GPU execution time
        torch::cuda::synchronize(); 

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        std::cout << "Pass " << i << " Time: " << duration.count() << " ms" << std::endl;
    }

    return 0;
}
