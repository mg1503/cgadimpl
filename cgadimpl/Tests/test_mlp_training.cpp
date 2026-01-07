// #include "ad/ag_all.hpp"
// #include <iostream>
// #include <cassert>
// #include <stdexcept>
// #include <vector>

// // --- A Simple MLP Model using nn::Sequential for clean composition ---
// class SimpleMLP : public ag::nn::Module {
// public:
//     ag::nn::Sequential layers;

//     // --- FIX #1: The constructor must accept ag::Device ---
//     SimpleMLP(Device device = Device::CPU) : 
//         layers({
//             new ag::nn::Linear(10, 32, device),
//             new ag::nn::ReLU(),
//             new ag::nn::Linear(32, 16, device),
//             new ag::nn::ReLU(),
//             new ag::nn::Linear(16, 5, device)
//         }) 
//     {
//         // Get the parameters from the sequential container
//         params_ = layers.parameters();
//     }

//     // This makes the model callable with a Tensor, e.g., model(x_tensor)
//     using ag::nn::Module::operator();

//     // --- FIX #2: The signature must be 'Value' not 'const Value&' to correctly override ---
//     ag::Value operator()(ag::Value x) override {
//         return layers(x);
//     }
// };

// // --- The Main Test Function ---
// int main() {
//     try {
//         // Test 1: CPU Training
//         std::cout << "========================================\n";
//         std::cout << "--- Starting End-to-End CPU Training ---\n";
//         std::cout << "========================================\n";

//         // This will now compile correctly
//         SimpleMLP model_cpu;
//         std::cout << "CPU Model created successfully.\n";

//         // --- FIX #3: Use ag::Device for options ---
//         auto cpu_opts = OwnTensor::TensorOptions().with_device(Device::CPU);
//         ag::Value input = ag::make_tensor(OwnTensor::Tensor::randn(OwnTensor::Shape{{8, 10}}, cpu_opts), "input");
//         ag::Value labels = ag::make_tensor(OwnTensor::Tensor::zeros(OwnTensor::Shape{{8, 5}}, cpu_opts), "labels");
        
//         model_cpu.zero_grad();
//         ag::Value output = model_cpu(input);
//         ag::Value loss = ag::mse_loss(output, labels);
//         ag::backward(loss);

//         float initial_loss_val = loss.val().to_cpu().data<float>()[0];
//         std::cout << "Initial Loss (CPU): " << initial_loss_val << std::endl;
        
//         const auto& w1_grad_cpu = model_cpu.parameters()[0].grad();
//         float grad_sum_cpu = OwnTensor::reduce_sum(OwnTensor::abs(w1_grad_cpu, nullptr)).to_cpu().data<float>()[0];
        
//         assert(grad_sum_cpu > 0.0f && "Gradients should not be zero after backward pass on CPU.");
//         std::cout << "PASS: Gradients were computed on CPU.\n";

//         float lr = 0.01f;
//         for (auto& param : model_cpu.parameters()) {
//             if (param.node && param.node->requires_grad()) {
//                 param.val() -= lr * param.grad();
//             }
//         }
        
//         ag::Value new_loss = ag::mse_loss(model_cpu(input), labels);
//         float new_loss_val = new_loss.val().to_cpu().data<float>()[0];
//         std::cout << "New Loss (CPU): " << new_loss_val << std::endl;

//         assert(new_loss_val < initial_loss_val && "Loss did not decrease after one training step on CPU.");
//         std::cout << "PASS: Loss decreased on CPU, indicating successful training step.\n";

        
//         // Test 2: GPU Training
//         if (!OwnTensor::device::cuda_available()) {
//             std::cout << "\nCUDA not available. Skipping GPU tests." << std::endl;
//             return 0;
//         }

//         std::cout << "\n========================================\n";
//         std::cout << "--- Starting End-to-End GPU Training ---\n";
//         std::cout << "========================================\n";
        
//         // --- FIX #3: Use ag::Device ---
//         SimpleMLP model_gpu(Device::CUDA);
//         std::cout << "GPU Model created successfully.\n";
        
//         auto gpu_opts = OwnTensor::TensorOptions().with_device(Device::CUDA);
//         ag::Value input_gpu = ag::make_tensor(OwnTensor::Tensor::randn(OwnTensor::Shape{{8, 10}}, gpu_opts), "input_gpu");
//         ag::Value labels_gpu = ag::make_tensor(OwnTensor::Tensor::zeros(OwnTensor::Shape{{8, 5}}, gpu_opts), "labels_gpu");
        
//         ag::Value loss_gpu = ag::mse_loss(model_gpu(input_gpu), labels_gpu);
//         ag::backward(loss_gpu);
//         cudaDeviceSynchronize();

//         float grad_sum_gpu = OwnTensor::reduce_sum(OwnTensor::abs(model_gpu.parameters()[0].grad(), nullptr)).to_cpu().data<float>()[0];
//         assert(grad_sum_gpu > 0.0f && "Gradients should not be zero after backward pass on GPU.");
//         std::cout << "PASS: Gradients were computed on GPU.\n";

//     } catch (const std::exception& e) {
//         std::cerr << "\nERROR: An exception occurred during the test: " << e.what() << std::endl;
//         return 1;
//     }

//     std::cout << "\nAll end-to-end training tests passed successfully!" << std::endl;
//     return 0;
// }
#include "ad/ag_all.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>

// Use the official ReLU module from the framework
using ReLU = ag::nn::ReLU;

int main() {
    try {
        std::cout << "========================================\n";
        std::cout << "--- Starting End-to-End CPU Training ---\n";
        std::cout << "========================================\n\n";

        const int batch_size = 8;
        const int in_features = 16;
        const int hidden_features = 32;
        const int out_features = 4;
        const float learning_rate = 0.01f;

        ag::nn::Sequential model_cpu({
            new ag::nn::Linear(in_features, hidden_features),
            new ReLU(),
            new ag::nn::Linear(hidden_features, out_features)
        });
        std::cout << "CPU Model created successfully.\n";

        auto cpu_opts = OwnTensor::TensorOptions().with_device(Device::CPU);
        ag::Value input = ag::make_tensor(OwnTensor::Tensor::randn<float>(OwnTensor::Shape{{batch_size, in_features}}, cpu_opts), "input");
        ag::Value labels = ag::make_tensor(OwnTensor::Tensor::zeros(OwnTensor::Shape{{batch_size, out_features}}, cpu_opts), "labels");
        
        // --- Initial Pass ---
        ag::Value output = model_cpu(input);
        ag::Value loss = ag::mse_loss(output, labels);
        ag::backward(loss);

        float initial_loss_val = loss.val().to_cpu().data<float>()[0];
        std::cout << "Initial Loss (CPU): " << initial_loss_val << std::endl;
        
        const auto& w1_grad_cpu = model_cpu.parameters()[0].grad();
        float grad_sum_cpu = OwnTensor::reduce_sum(OwnTensor::abs(w1_grad_cpu, nullptr)).to_cpu().data<float>()[0];
        
        assert(grad_sum_cpu > 0.0f && "FATAL: Gradients are zero after first backward pass!");
        std::cout << "PASS: Gradients were computed successfully on CPU.\n";

        // --- SGD Step ---
        for (auto& param : model_cpu.parameters()) {
            if (param.node && param.node->requires_grad()) {
                param.val() -= learning_rate * param.grad();
            }
        }
        std::cout << "Performed one SGD step.\n";
        
        // --- Second Pass ---
        model_cpu.zero_grad();
        ag::Value new_output = model_cpu(input);
        ag::Value new_loss = ag::mse_loss(new_output, labels);
        float new_loss_val = new_loss.val().to_cpu().data<float>()[0];
        std::cout << "New Loss (CPU): " << new_loss_val << std::endl;

        assert(new_loss_val < initial_loss_val && "FATAL: Loss did not decrease after training step!");
        std::cout << "PASS: Loss decreased on CPU, indicating successful training.\n";

    } catch (const std::exception& e) {
        std::cerr << "\nERROR: An exception occurred during the test: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n✅ All end-to-end training tests passed successfully!\n";
    return 0;
}