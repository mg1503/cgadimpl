#include "ad/ag_all.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>

using namespace ag;
using namespace OwnTensor;

// ==========================================================
// A manual, raw forward function for our "Kitchen Sink" MLP
// It takes a list of parameters and constructs the graph.
// ==========================================================
Value kitchen_sink_forward(Value x, std::vector<Value>& params) {
    // Layer 1: Linear + Custom Activations
    Value l1 = linear(x, params[0], params[1]); // W1, b1
    Value a1 = laynor(l1);
    Value a2 = gelu(a1);
    Value a3 = silu(a2);
    
    // For this raw test, we'll use a fixed alpha for leaky_relu
    Value a4 = leaky_relu(a3, 0.01f);
    Value a5 = tanh(a4);
    Value a6 = rms(a5);

    // Layer 2: Final Linear Layer
    Value logits = linear(a6, params[2], params[3]); // W2, b2

    return logits;
}

// ==========================================================
// The Main Training Test
// ==========================================================
void test_manual_mlp_training() {
    std::cout << "\n==================================================\n";
    std::cout << "--- Manual MLP Training Integration Test ---\n";
    std::cout << "==================================================\n";

    #ifndef WITH_CUDA
        std::cout << "Test skipped: Not compiled with CUDA support.\n";
        return;
    #endif

    const int batch_size = 16;
    const int in_features = 64;
    const int hidden_features = 32;
    const int out_features = 10;
    const int epochs = 5;
    const float learning_rate = 0.01f;
    Device dev = Device::CUDA;
    auto opts = TensorOptions().with_device(dev).with_req_grad(true);

    // --- 1. Manually Create All Parameters ---
    std::vector<Value> parameters;
    // Layer 1
    parameters.push_back(make_tensor(Tensor::randn<float>(Shape{{hidden_features, in_features}}, opts) * 0.1, "W1"));
    parameters.push_back(make_tensor(Tensor::zeros(Shape{{1, hidden_features}}, opts), "b1"));
    // Layer 2
    parameters.push_back(make_tensor(Tensor::randn<float>(Shape{{out_features, hidden_features}}, opts) * 0.1, "W2"));
    parameters.push_back(make_tensor(Tensor::zeros(Shape{{1, out_features}}, opts), "b2"));
    
    std::cout << "Manually created " << parameters.size() << " parameter tensors on CUDA.\n";

    // --- 2. Create Data ---
    Tensor x_data = Tensor::randn<float>(Shape{{batch_size, in_features}}, TensorOptions().with_device(dev));
    Tensor y_target_data = Tensor::randn<float>(Shape{{batch_size, out_features}}, TensorOptions().with_device(dev));
    
    Value x = make_tensor(x_data);
    Value y_target = make_tensor(y_target_data);

    // --- 3. The Training Loop ---
    double initial_loss = -1.0;
    double final_loss = -1.0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // a. Zero out all gradients from the previous iteration.
        for (Value& p : parameters) {
            zero_grad(p);
        }

        // b. Forward pass: call our raw function.
        Value y_pred = kitchen_sink_forward(x, parameters);

        // c. Compute the loss.
        Value loss = mse_loss(y_pred, y_target);

        // Dump graph visualization only once on first epoch
        if (epoch == 0) {
            ag::debug::dump_dot(loss, "graph_loss.jpg");
        }
        
        // d. Backward pass: compute gradients for all parameters.
        backward(loss);

        // e. Optimizer step: update all parameters using their gradients.
        for (Value& param : parameters) {
            // Manual SGD update, directly modifying the node's value tensor
            param.node->value -= (param.grad() * learning_rate);
        }

        double current_loss = loss.val().to_cpu().data<float>()[0];
        if (epoch == 0) initial_loss = current_loss;
        final_loss = current_loss;

        std::cout << "Epoch " << epoch << ", Loss: " << std::fixed << std::setprecision(4) << current_loss << std::endl;
    }
     

    // --- 4. Validation ---
    std::cout << "Initial Loss: " << initial_loss << ", Final Loss: " << final_loss << std::endl;
    assert(final_loss < initial_loss);
    assert(!std::isnan(final_loss) && !std::isinf(final_loss));

    std::cout << "\n✅ Training successful. Loss decreased, confirming core graph and autodiff functionality.\n";
}

int main() {
    try {
        test_manual_mlp_training();
    } catch (const std::exception& e) {
        std::cerr << "\nCaught exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}