#include <torch/torch.h>
#include <iostream>
#include <chrono>

struct ManualMLP : torch::nn::Module {
    ManualMLP(int64_t in_dim, int64_t hid_dim, int64_t out_dim) {
        // Define weights and biases manually for the DAG
        w1 = register_parameter("w1", torch::randn({in_dim, hid_dim}));
        b1 = register_parameter("b1", torch::randn({hid_dim}));
        w2 = register_parameter("w2", torch::randn({hid_dim, out_dim}));
        b2 = register_parameter("b2", torch::randn({out_dim}));
    }

    // DAG implementation using matmul and add
    torch::Tensor forward(torch::Tensor x) {
        x = torch::add(torch::matmul(x, w1), b1);
        x = torch::relu(x);
        x = torch::add(torch::matmul(x, w2), b2);
        return x;
    }

    torch::Tensor w1, b1, w2, b2;
};

int main() {
    // 2025 CUDA 13.0 Setup
    torch::Device device(torch::kCUDA);
    
    // Initialize MLP: Input 784 -> Hidden 1024 -> Output 10
    auto model = std::make_shared<ManualMLP>(784, 1024, 10);
    model->to(device);

    // Dummy batch of 32 images
    torch::Tensor input = torch::randn({32, 784}).to(device);

    // Warm-up pass to initialize CUDA kernels
    {
        torch::NoGradGuard no_grad;
        model->forward(input);
        torch::cuda::synchronize();
    }

    std::cout << "Starting 5 sequential forward passes on RTX 3090..." << std::endl;

    for (int i = 1; i <= 5; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        // Perform forward pass
        torch::Tensor output = model->forward(input);

        // Synchronize is required for accurate GPU timing
        torch::cuda::synchronize(); 

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        std::cout << "Pass " << i << " | Time: " << duration.count() << " ms" << std::endl;
    }

    return 0;
}
