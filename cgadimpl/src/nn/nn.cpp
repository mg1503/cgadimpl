// ===================================================================
// file: cgadimpl/src/nn/nn.cpp (Corrected and Modernized)
// ===================================================================
#include "nn/nn.hpp"
#include <cmath>
#include <cassert>

// Note: "tensor.hpp" is now the adapter that includes the full OwnTensor library
#include "tensor.hpp" 

namespace ag::nn {

// --- Module and Linear Class Implementations ---

void Module::to(Device dev) {
    for (Value& p : params_) {
        if (p.node) {
            // .to() method is correct.
            p.node->value = p.node->value.to(dev);

            // FIX: Use the new OwnTensor::Tensor::zeros factory.
            // We pass the new shape and options to ensure the grad tensor
            // is created on the correct device with the correct dtype.
            p.node->grad = OwnTensor::Tensor::zeros(p.node->value.shape(), ag::options(p.node->value));
        }
    }
}

void Module::zero_grad() {
    for (Value& p : params_) {
        // We can reuse the same logic as in Module::to
        if (p.node && p.node->requires_grad()) {
            p.node->grad = OwnTensor::Tensor::zeros(p.node->value.shape(), ag::options(p.node->value));
        }
    }
}

Linear::Linear(int in_features, int out_features, Device dev) {
    // "Kaiming He Initialization" (or just "He Initialization").
    float scale = sqrtf(2.0f / in_features);

    // FIX: Use the new TensorOptions and factory functions.
    // 1. Create an options object that specifies the device and that these are trainable parameters.
    auto param_opts = OwnTensor::TensorOptions()
                        .with_device(dev)
                        .with_req_grad(true);

    // 2. Call the new factory functions with a Shape struct {} and the options object.
    Tensor w_tensor = OwnTensor::Tensor::randn(Shape{{out_features, in_features}}, param_opts) * scale;
    Tensor b_tensor = OwnTensor::Tensor::zeros(Shape{{1, out_features}}, param_opts);

    // 3. Call the simplified make_tensor. It infers requires_grad from the tensor itself.
    W = make_tensor(w_tensor, "W");
    b = make_tensor(b_tensor, "b");

    params_.push_back(W);
    params_.push_back(b);
}

// Forward pass definition (no changes needed, this is correct)
Value Linear::operator()(const Value& input) {   
    return linear(input, W, b);
}

// FIX: Corrected a C++ bug. The 'modules' vector must be used to initialize 'layers_'.
Sequential::Sequential(const std::vector<Module*>& modules) : layers_(modules) {
    // Now that layers_ is initialized, this loop is correct.
    for (auto* mod : layers_) {
        params_.insert(params_.end(), mod->parameters().begin(), mod->parameters().end());
    }
}

// This is correct, no changes needed.
Value Sequential::operator()(Value x) {
    for (auto* layer : layers_) {
        x = (*layer)(x); // Pass the output of one layer to the next
    }
    return x;
}

// This is correct, no changes needed.
Value ReLU::operator()(const Value& input) {
    return ag::relu(input); // Just calls the high-level op.
}

} // namespace ag::nn