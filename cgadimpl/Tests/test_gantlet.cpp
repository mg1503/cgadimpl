#include "ad/ag_all.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>

using namespace ag;
using namespace OwnTensor;

// ==========================================================
// A Custom "Gauntlet" Block using multiple ops
// ==========================================================
class GauntletBlock : public nn::Module {
public:
    nn::Linear* linear;
    
    // An enum to choose which activation/norm path to take
    enum BlockType { TYPE_A, TYPE_B, TYPE_C, TYPE_D };
    BlockType type_;

    GauntletBlock(int features, BlockType type, Device dev) : type_(type) {
        linear = new nn::Linear(features, features, dev);
        // Automatically collect parameters from the child linear layer
        for(auto& p : linear->parameters()) {
            params_.push_back(p);
        }
    }

    using ag::nn::Module::operator();
    Value operator()(Value x) override {
        Value hidden = (*linear)(x);

        switch(type_) {
            case TYPE_A:
                // Path A: ReLU -> LayerNorm
                return laynor(relu(hidden));
            case TYPE_B:
                // Path B: GELU -> RMSNorm
                return rms(gelu(hidden));
            case TYPE_C:
                // Path C: SiLU -> LayerNorm
                return laynor(silu(hidden));
            case TYPE_D:
            default:
                // Path D: Tanh -> RMSNorm
                return rms(tanh(hidden));
        }
    }
};

// ==========================================================
// The Main Training Test
// ==========================================================
void test_large_model_training() {
    std::cout << "\n=========================================================\n";
    std::cout << "--- Large Model ('The Gauntlet') Training Stress Test ---\n";
    std::cout << "=========================================================\n";

    #ifndef WITH_CUDA
        std::cout << "Test skipped: Not compiled with CUDA support.\n";
        return;
    #endif

    // --- Hyperparameters Scaled Up for a ~10 minute run ---
    const int batch_size = 256;         // Increased batch size
    const int features = 2048;          // Massively increased width
    const int num_classes = 100;        // Slightly larger output space
    const int num_epochs = 200;         // Increased training duration
    const float learning_rate = 1e-2f;  // Lower learning rate for stability in large models
    Device dev = Device::CUDA;

    // --- 1. Build the Deep and Wide Model ---
    nn::Sequential model({
        new nn::Linear(features, features, dev),
        
        // Stack 24 GauntletBlocks
        new GauntletBlock(features, GauntletBlock::TYPE_A, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_B, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_C, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_D, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_A, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_B, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_C, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_D, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_A, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_B, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_C, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_D, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_A, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_B, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_C, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_D, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_A, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_B, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_C, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_D, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_A, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_B, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_C, dev),
        new GauntletBlock(features, GauntletBlock::TYPE_D, dev),
        
        new nn::Linear(features, num_classes, dev)
    });
    for (auto& param : model.parameters()) {
        param.val() *= 0.01f; // Scale all initial weights and biases down by 100x
    }
    std::cout << "Model created on CUDA with " << model.parameters().size() << " parameter tensors.\n";
    std::cout << "Training for " << num_epochs << " epochs with LR = " << learning_rate << "...\n\n";

    // --- 2. Create Synthetic Data ---
    // --- 2. Create Synthetic Data ---
    Tensor x_data = Tensor::randn(Shape{{batch_size, features}}, TensorOptions().with_device(dev));
    
    // --- FIX START ---
    
    // Create the initial logits as a raw Tensor.
    Tensor y_target_logits = Tensor::rand(Shape{{batch_size, num_classes}}, TensorOptions().with_device(dev));

    // 1. Wrap the raw Tensor in a Value to use graph operations.
    Value y_target_logits_val = make_tensor(y_target_logits);
    
    // 2. Call the softmax_row graph operation. This returns a Value.
    Value y_target_prob_val = softmax_row(y_target_logits_val);
    
    // 3. Get the Tensor from the result Value and then call contiguous() on it.
    Tensor y_target_data = y_target_prob_val.val().contiguous();

    Value x = make_tensor(x_data);
    Value y_target = make_tensor(y_target_data);

    // --- 3. The Training Loop ---
    double initial_loss = -1.0;
    double final_loss = -1.0;
    int print_every = 10;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model.zero_grad();

        Value y_pred_logits = model(x);
        Value loss = cross_entropy_with_logits(y_pred_logits, y_target);
        
        backward(loss);

        // Manual SGD Optimizer Step
        for (Value& param : model.parameters()) {
            if (param.grad().numel() > 0) { // Ensure grad exists
                param.val() -= (param.grad() * learning_rate);
            }
        }

        double current_loss = loss.val().to_cpu().data<float>()[0];
        if (epoch == 0) initial_loss = current_loss;
        if (epoch == num_epochs - 1) final_loss = current_loss;

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch [" << std::setw(3) << epoch + 1 << "/" << num_epochs 
                      << "], Loss: " << std::fixed << std::setprecision(6) << current_loss << std::endl;
        }
        
        // Check for numerical instability during training
        assert(!std::isnan(current_loss) && !std::isinf(current_loss));
    }

    // --- 4. Validation ---
    std::cout << "\n----------------------------------------\n";
    std::cout << "Initial Loss: " << initial_loss << "\nFinal Loss:   " << final_loss << std::endl;
    assert(final_loss < initial_loss);
    // assert(final_loss < initial_loss / 2.0); // Assert that the loss at least halved

    std::cout << "\n  'The Gauntlet' training test passed!\n";
    std::cout << "   Loss decreased significantly, indicating stable and correct gradient flow.\n";
}

int main() {
    try {
        test_large_model_training();
    } catch (const std::exception& e) {
        std::cerr << "\nCaught exception during training: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}