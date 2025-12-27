#include <torch/torch.h>
#include <iostream>

struct ManualMLP : torch::nn::Module {
    ManualMLP(int64_t input_dim, int64_t hidden_dim, int64_t output_dim) {
        // 1. Initialize weights and biases using torch::randn
        // 2. Register them as parameters so the optimizer can find them
        w1 = register_parameter("w1", torch::randn({input_dim, hidden_dim}));
        b1 = register_parameter("b1", torch::randn(hidden_dim));
        
        w2 = register_parameter("w2", torch::randn({hidden_dim, output_dim}));
        b2 = register_parameter("b2", torch::randn(output_dim));
    }

    // Explicitly define the DAG functionality in the forward pass
    torch::Tensor forward(torch::Tensor x) {
        // Layer 1: y = (x * w1) + b1
        // Use torch::matmul for matrix multiplication
        // Use torch::add for adding bias
        x = torch::add(torch::matmul(x, w1), b1);
        x = torch::relu(x); // Activation function

        // Layer 2: y = (x * w2) + b2
        x = torch::add(torch::matmul(x, w2), b2);
        
        return x;
    }

    torch::Tensor w1, b1, w2, b2;
};

int main() {
    // Instantiate the model
    int64_t in = 784, hid = 128, out = 10;
    auto model = std::make_shared<ManualMLP>(in, hid, out);

    // Create a dummy input tensor [Batch Size: 8, Features: 784]
    torch::Tensor input = torch::randn({8, 784});

    // Perform forward pass
    torch::Tensor output = model->forward(input);

    std::cout << "Output shape: " << output.sizes() << std::endl;
    return 0;
}
