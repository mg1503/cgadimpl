#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include "ad/ag_all.hpp"
#include "ad/optimizer/optim.hpp"

using namespace ag;
using namespace OwnTensor;

// Helper to initialize a tensor with random normal values
void init_weight(Tensor& t, float std_dev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std_dev);
    float* data = t.data<float>();
    size_t n = t.numel();
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(gen);
    }
}

// Helper to initialize bias with zeros
void init_bias(Tensor& t) {
    t.fill(0.0f);
}

int main() {
    std::cout << "===== Adam Optimizer Test: 10-Layer MLP Classification =====\n";

    // 1. Problem Dimensions
    const int batch_size = 16;
    const int input_dim = 128;
    const int hidden_dim = 128;
    const int output_dim = 2; // Binary classification (one-hot)
    const int num_hidden_layers = 10;

    // 2. Generate Synthetic Data
    // X: random inputs
    Tensor x_data(Shape{{batch_size, input_dim}}, false);
    init_weight(x_data, 1.0f);
    Value X = make_tensor(x_data, "input");

    // Y: target labels (one-hot)
    Tensor y_data(Shape{{batch_size, output_dim}}, false);
    for (int i = 0; i < batch_size; ++i) {
        // Simple rule: if sum of inputs > 0, category 1, else category 0
        float sum = 0;
        float* row = x_data.data<float>() + i * input_dim;
        for (int j = 0; j < input_dim; ++j) sum += row[j];
        
        if (sum > 0) {
            y_data.data<float>()[i * output_dim + 0] = 0.0f;
            y_data.data<float>()[i * output_dim + 1] = 1.0f;
        } else {
            y_data.data<float>()[i * output_dim + 0] = 1.0f;
            y_data.data<float>()[i * output_dim + 1] = 0.0f;
        }
    }
    Value Y = make_tensor(y_data, "target");

    // 3. Define MLP Parameters
    std::vector<Value> params;
    
    struct Layer {
        Value W;
        Value b;
    };
    std::vector<Layer> layers;

    // input to first hidden
    {
        Tensor w_t(Shape{{hidden_dim, input_dim}}, TensorOptions().with_req_grad(true));
        init_weight(w_t, std::sqrt(1.0f / input_dim));
        Value W = make_tensor(w_t, "W_in");
        
        Tensor b_t(Shape{{1, hidden_dim}}, TensorOptions().with_req_grad(true));
        init_bias(b_t);
        Value b = make_tensor(b_t, "b_in");
        
        layers.push_back({W, b});
        params.push_back(W);
        params.push_back(b);
    }

    // internal hidden layers
    for (int i = 0; i < num_hidden_layers - 1; ++i) {
        Tensor w_t(Shape{{hidden_dim, hidden_dim}}, TensorOptions().with_req_grad(true));
        init_weight(w_t, std::sqrt(1.0f / hidden_dim));
        Value W = make_tensor(w_t, ("W_h" + std::to_string(i)).c_str());
        
        Tensor b_t(Shape{{1, hidden_dim}}, TensorOptions().with_req_grad(true));
        init_bias(b_t);
        Value b = make_tensor(b_t, ("b_h" + std::to_string(i)).c_str());
        
        layers.push_back({W, b});
        params.push_back(W);
        params.push_back(b);
    }

    // last hidden to output
    {
        Tensor w_t(Shape{{output_dim, hidden_dim}}, TensorOptions().with_req_grad(true));
        init_weight(w_t, std::sqrt(1.0f / hidden_dim));
        Value W = make_tensor(w_t, "W_out");
        
        Tensor b_t(Shape{{1, output_dim}}, TensorOptions().with_req_grad(true));
        init_bias(b_t);
        Value b = make_tensor(b_t, "b_out");
        
        layers.push_back({W, b});
        params.push_back(W);
        params.push_back(b);
    }

    // 4. Initialize Adam Optimizer
    Adam optimizer(params, 0.001f);

    // 5. Training Loop
    std::cout << "Starting Training (20 iterations)...\n";
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int i = 0; i < 20; ++i) {
        // Forward Pass
        Value hidden = X;
        for (size_t l = 0; l < layers.size() - 1; ++l) {
            hidden = ag::tanh(ag::linear(hidden, layers[l].W, layers[l].b));
        }
        // Output Layer (Logits)
        Value logits = ag::linear(hidden, layers.back().W, layers.back().b);
        
        // Loss
        Value loss = ag::cross_entropy_with_logits(logits, Y);
        
        // Backward Pass
        optimizer.zero_grad();
        backward(loss);
        
        float current_loss = loss.val().data<float>()[0];
        if (i == 0) initial_loss = current_loss;
        final_loss = current_loss;

        std::cout << "Iteration " << std::setw(2) << i + 1 
                  << " | Loss: " << std::fixed << std::setprecision(6) << current_loss << std::endl;
        
        // Update
        optimizer.step();
    }

    std::cout << "\nSummary:\n";
    std::cout << "Initial Loss: " << initial_loss << "\n";
    std::cout << "Final Loss:   " << final_loss << "\n";

    if (final_loss < initial_loss) {
        std::cout << "  SUCCESS: Loss decreased successfully.\n";
    } else {
        std::cout << "❌ FAILURE: Loss did not decrease.\n";
    }

    return 0;
}