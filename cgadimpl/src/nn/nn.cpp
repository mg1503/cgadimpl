// =============================================
// cgadimpl/src/nn/nn.cpp
// =============================================
#include "nn/nn.hpp"
#include <cmath>
#include <cassert>
#include "tensor.hpp"

namespace ag::nn {


// --- NEW: Module and Linear Class Implementations ---

void Module::to(Device dev) {
    for (Value& p : params_) {
        if (p.node) {
            p.node->value = p.node->value.to(dev);
            p.node->grad = Tensor::zeros_like(p.node->value);
        }
    }
}

void Module::zero_grad() {
    for (Value& p : params_) {
        if (p.node && p.node->requires_grad) {
            p.node->grad = Tensor::zeros_like(p.node->value);
        }
    }
}

Linear::Linear(int in_features, int out_features, Device dev) {
    float scale = sqrtf(2.0f / in_features);
    Tensor w_tensor = Tensor::randn(out_features , in_features, 42, dev) * scale;
    Tensor b_tensor = Tensor::zeros(1, out_features, dev);

    W = make_tensor(w_tensor, "W", true);
    b = make_tensor(b_tensor, "b", true);

    params_.push_back(W);
    params_.push_back(b);
}

// Forward pass definition
Value Linear::operator()(const Value& input) {   
    return linear(input, W, b);
}

Sequential::Sequential(const std::vector<Module*>& modules) {
     // The constructor now does two important things:
    // 1. It saves the list of layers into the `layers_` member variable (see the `: layers_(modules)` part).
    // 2. It iterates through those saved layers to collect all of their parameters.
    for (auto* mod : layers_) {
        params_.insert(params_.end(), mod->parameters().begin(), mod->parameters().end());
    }
}

Value Sequential::operator()(Value x) {
    for (auto* layer : layers_) {
        x = (*layer)(x); // Pass the output of one layer to the next
    }
    return x;
}

Value ReLU::operator()(const Value& input) {
    return ag::relu(input); // Just calls the op. It has no parameters.
}

} // namespace ag::nn







// Tensor silu(const Tensor& x){
//     Tensor y(x.rows(), x.cols());
//     for(int i=0;i<x.rows();++i) for(int j=0;j<x.cols();++j){
//         float v=x(i,j); float s=1.f/(1.f+std::exp(-v)); y(i,j)=v*s;
//     }
//     return y;
// }
// Tensor gelu(const Tensor& x){
//     Tensor y(x.rows(), x.cols());
//     constexpr float c = 0.7978845608028654f; // sqrt(2/pi)
//     for(int i=0;i<x.rows();++i) for(int j=0;j<x.cols();++j){
//         float v=x(i,j);
//         float u=c*(v+0.044715f*v*v*v);
//         y(i,j)=0.5f*v*(1.f+std::tanh(u));
//     }
//     return y;
// }
