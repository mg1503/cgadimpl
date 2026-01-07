// #include "nn/nn.hpp"
// #include "ad/ag_all.hpp"
// #include <iostream>
// #include <vector> // Required for y.shape()

// int main() {
//     using namespace ag;
//     using namespace OwnTensor; // To easily access Shape, TensorOptions, etc.

//     // 1. Create a linear layer on the CPU
//     nn::Linear fc1(128, 64, Device::CPU);

//     // 2. Move the entire layer (weights and biases) to the GPU
//     std::cout << "Moving model to GPU..." << std::endl;
//     fc1.to(Device::CUDA);

//     // 3. Create some input data directly on the GPU
//     // --- FIX START ---
//     // Use the modern factory function with Shape and TensorOptions
//     Tensor x_tensor = Tensor::randn(Shape{{10, 128}}, TensorOptions().with_device(Device::CUDA));
//     // Use the standard make_tensor factory. 'requires_grad' is false by default.
//     Value x = make_tensor(x_tensor, "input");
//     // --- FIX END ---

//     // 4. Perform a forward pass on the GPU
//     std::cout << "Performing forward pass on GPU..." << std::endl;
//     Value y = fc1(x);

//     // 5. Verify and print results
//     // --- FIX START ---
//     // Use modern shape access
//     const std::vector<int64_t>& output_shape = y.shape();
//     std::cout << "Output is on CUDA: " << (y.val().is_cuda() ? "true" : "false") << std::endl;
//     std::cout << "Output shape: " << output_shape[0] << "x" << output_shape[1] << std::endl;
//     // --- FIX END ---

//     // You can now proceed with loss calculation and backpropagation
//     // For example:
//     // Tensor targets = Tensor::randn(Shape{{10, 64}}, TensorOptions().with_device(Device::CUDA));
//     // Value loss = mse_loss(y, make_tensor(targets));
//     // backward(loss);

//     std::cout << "\nTest completed successfully." << std::endl;

//     return 0;
// }
#include "ad/ag_all.hpp" // Main umbrella header for the framework
#include <iostream>
#include <vector>
#include <iomanip>


int main() {
    using namespace ag;
    using namespace OwnTensor;

    std::cout << "========================================\n";
    std::cout << "--- Starting End-to-End MLP Training ---\n";
    std::cout << "========================================\n\n";

    // 1. --- Define Hyperparameters ---
    const int batch_size = 16;
    const int input_features = 128;
    const int hidden_features = 64;
    const int output_features = 10;
    const float learning_rate = 0.01f;
    const int epochs = 15;

    // 2. --- Create the Model ---
    ag::nn::Sequential model({
        new ag::nn::Linear(input_features, hidden_features),
        new ag::nn::ReLU(),
        new ag::nn::Linear(hidden_features, output_features)
    });
    std::cout << "Model created with " << model.parameters().size() << " parameter tensors.\n\n";

    // 3. --- Generate Random Data ---
    // Tensor x_tensor = 
    Tensor y_tensor = Tensor::randn<float>(Shape{{batch_size, output_features}}, TensorOptions().with_req_grad(true));
    Value X = make_tensor(Tensor::randn<float>(Shape{{batch_size, input_features}}, TensorOptions().with_req_grad(true)), "X_data");
    Value Y = make_tensor(y_tensor, "Y_target");

    // 4. --- The Training Loop ---
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Value predictions = model(X);
        Value loss = mse_loss(predictions, Y);

        float loss_value = loss.val().data<float>()[0];
        std::cout << "Epoch " << std::setw(2) << epoch 
                  << ", Loss: " << std::fixed << std::setprecision(4) << loss_value << std::endl;

        model.zero_grad();
        backward(loss);

        // This loop will now work because Module::parameters() is non-const
        for (Value& param : model.parameters()) {
            if (param.node && param.node->requires_grad()) {
                // This += will now work because param.val() is non-const
                param.val() += (param.grad() * -learning_rate);
            }
        }
    }

    // 5. --- Clean up dynamically allocated modules ---
    // This will now work because get_layers() exists
    for (auto* layer : model.get_layers()) {
        delete layer;
    }

    std::cout << "\n✅ Training finished successfully.\n";
    return 0;
}








