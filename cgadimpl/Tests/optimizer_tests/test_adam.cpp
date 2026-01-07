#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include "ad/ag_all.hpp"
#include "optim.hpp"

using namespace ag;
using namespace OwnTensor;

// No separate initialization functions needed - will use GPU tensor operations directly

int main() {
    // Device selection
    #ifdef WITH_CUDA
        Device dev = Device::CUDA;
        std::cout << "===== Adam Optimizer Test 2: High-Dimensional MLP (1024) with Mixed Activations (GPU) =====\n";
        std::cout << "Running on: CUDA GPU\n";
    #else
        Device dev = Device::CPU;
        std::cout << "===== Adam Optimizer Test 2: High-Dimensional MLP (1024) with Mixed Activations (CPU) =====\n";
        std::cout << "Running on: CPU\n";
    #endif
    
    auto opts = TensorOptions().with_device(dev);

    // 1. Dimensions
    const int batch_size = 8;
    const int dim = 1024;
    const int num_iterations = 20;

    // 2. Data Generation - All on GPU!
    // Use GPU randn for random initialization
    Tensor x_data = Tensor::randn<float>(Shape{{batch_size, dim}}, opts);
    Value X = make_tensor(x_data, "input");

    // Target is just 2 * input + 0.5 (simple linear target) - computed on GPU
    Tensor y_data = x_data * 2.0f + 0.5f;
    Value Y = make_tensor(y_data, "target");

    // 3. Define MLP with Mixed Activations
    std::vector<Value> params;
    
    auto create_layer = [&](int in_d, int out_d, const std::string& name) {
        // Initialize weights with GPU randn and scale by std_dev
        float std_dev = std::sqrt(2.0f / in_d);
        Tensor w_t = Tensor::randn<float>(Shape{{out_d, in_d}}, TensorOptions().with_device(dev).with_req_grad(true)) * std_dev;
        Value W = make_tensor(w_t, (name + "_W").c_str());
        
        // Initialize bias on GPU with fill
        Tensor b_t = Tensor::zeros(Shape{{1, out_d}}, TensorOptions().with_device(dev).with_req_grad(true));
        b_t.fill(0.01f);
        Value b = make_tensor(b_t, (name + "_b").c_str());
        
        params.push_back(W);
        params.push_back(b);
        return std::make_pair(W, b);
    };

    auto L1 = create_layer(dim, dim, "L1");
    auto L2 = create_layer(dim, dim, "L2");
    auto L3 = create_layer(dim, dim, "L3");
    auto L4 = create_layer(dim, dim, "L4");
    auto L5 = create_layer(dim, dim, "L5");

    // 4. Initialize Adam
    Adam optimizer(params, 0.001f);

    std::cout << "Starting training loop with mixed activations (ReLU, Mish, GELU, SiLU, Tanh)...\n";

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int i = 0; i < num_iterations; ++i) {
        // Forward Pass with different activations
        Value h1 = ag::relu(ag::linear(X, L1.first, L1.second));
        Value h2 = ag::mish(ag::linear(h1, L2.first, L2.second));
        Value h3 = ag::gelu(ag::linear(h2, L3.first, L3.second));
        Value h4 = ag::silu(ag::linear(h3, L4.first, L4.second));
        Value out = ag::linear(h4, L5.first, L5.second); // Output linear (regression)

        // Loss
        Value loss = ag::mse_loss(out, Y);

        // Backward
        optimizer.zero_grad();
        backward(loss);

        // Update
        optimizer.step();

        // Transfer loss to CPU for printing
        float current_loss = (dev == Device::CUDA) ? loss.val().to_cpu().data<float>()[0] : loss.val().data<float>()[0];
        if (i == 0) initial_loss = current_loss;
        final_loss = current_loss;

        std::cout << "Iteration " << std::setw(2) << i + 1 
                  << " | Loss: " << std::fixed << std::setprecision(6) << current_loss << std::endl;
    }

    std::cout << "\nResults for 1024-dim MLP:\n";
    std::cout << "Initial Loss: " << initial_loss << "\n";
    std::cout << "Final Loss:   " << final_loss << "\n";

    if (final_loss < initial_loss) {
        std::cout << "  SUCCESS: Loss decreased with mixed activations.\n";
    } else {
        std::cout << "   FAILURE: Loss did not decrease.\n";
    }

    return 0;
}