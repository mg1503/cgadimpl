#include "nn/nn.hpp"
#include <cmath>
#include <cassert>
#include "tensor.hpp" 

namespace ag::nn {

void Module::to(DeviceIndex dev) {
    for (Value& p : params_) {
        if (p.node) {
            p.node->value = p.node->value.to(dev);
            p.node->grad = OwnTensor::Tensor::zeros(p.node->value.shape(), ag::options(p.node->value));
        }
    }
}

void Module::zero_grad() {
    for (Value& p : params_) {
        if (p.node && p.node->requires_grad()) {
            p.node->grad = OwnTensor::Tensor::zeros(p.node->value.shape(), ag::options(p.node->value));
        }
    }
}

Linear::Linear(int in_features, int out_features, Device dev) {
    float scale = sqrtf(2.0f / in_features);
    auto param_opts = OwnTensor::TensorOptions().with_device(dev).with_req_grad(true);
    Tensor w_tensor = OwnTensor::Tensor::randn<float>(Shape{{out_features, in_features}}, param_opts) * scale;
    Tensor b_tensor = OwnTensor::Tensor::zeros(Shape{{1, out_features}}, param_opts);
    W = make_tensor(w_tensor, "W");
    b = make_tensor(b_tensor, "b");
    params_.push_back(W);
    params_.push_back(b);
}

Value Linear::operator()(Value input) {   
    return linear(input, W, b);
}

Sequential::Sequential(const std::vector<Module*>& modules) : layers_(modules) {
    for (auto* mod : layers_) {
        for(auto& p : mod->parameters()) {
            params_.push_back(p);
        }
    }
}

Value Sequential::operator()(Value x) {
    for (auto* layer : layers_) {
        x = (*layer)(x);
    }
    return x;
}

Value ReLU::operator()(Value input) {
    return ag::relu(input);
}

} // namespace ag::nn