// ===================================================================
// file: cgadimpl/src/optim.cpp (Corrected for OwnTensor)
// ===================================================================
#include "ad/optimizer/optim.hpp"
#include <math.h>

// No new includes are needed because tensor.hpp brings in everything.

namespace ag {

void SGD(const Value& root, const Tensor* grad_seed, float learning_rate) {
    auto order = topo_from(root.node.get());
    for (Node* n : order) {
        if (n->op == Op::Leaf && n->requires_grad()) {
            n->value += -learning_rate * n->grad;
        }
    }
}

SGDOptimizer::SGDOptimizer(const std::vector<Value>& params, float learning_rate)
    : params_(params), learning_rate_(learning_rate) {
    for (const auto& p : params_) {
        Node* n = p.node.get();
        if (n->requires_grad() && n->value.dtype() != Dtype::Float32) {
            master_params_[n] = n->value.as_type(Dtype::Float32);
        }
    }
}

void SGDOptimizer::step() {
    for (const auto& p : params_) {
        Node* n = p.node.get();
        if (!n->requires_grad()) continue;

        Tensor grad_f32 = n->grad;
        if (grad_f32.dtype() != Dtype::Float32) {
            grad_f32 = grad_f32.as_type(Dtype::Float32);
        }

        if (master_params_.count(n)) {
            Tensor& master_p = master_params_[n];
            master_p += -learning_rate_ * grad_f32;
            n->value.copy_(master_p.as_type(n->value.dtype()));
        } else {
            n->value += -learning_rate_ * grad_f32;
        }
    }
}

void SGDOptimizer::zero_grad() {
    for (const auto& p : params_) {
        Node* n = p.node.get();
        if (n->requires_grad() && n->grad.is_valid()) {
            n->grad = Tensor::zeros(n->grad.shape(), TensorOptions().with_dtype(n->grad.dtype()).with_device(n->grad.device()));
        }
    }
}

const Tensor* SGDOptimizer::get_master_weight(const Value& v) const {
    auto it = master_params_.find(v.node.get());
    if (it != master_params_.end()) {
        return &it->second;
    }
    return nullptr;
}

Adam::Adam(const std::vector<Value>& params, float alpha, float beta1, float beta2, float epsilon)
    : params_(params), alpha_(alpha), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {
    
    for (const auto& p : params_) {
        Node* n = p.node.get();
        if (n->requires_grad()) {
            // Initialize moments and master parameters in Float32
            TensorOptions opts_f32 = options(n->value).with_dtype(Dtype::Float32);
            
            m_[n] = Tensor::zeros(n->value.shape(), opts_f32);
            v_[n] = Tensor::zeros(n->value.shape(), opts_f32);
            
            // If the parameter is not Float32, we need a master copy in Float32
            if (n->value.dtype() != Dtype::Float32) {
                master_params_[n] = n->value.as_type(Dtype::Float32);
            }
        }
    }
}

void Adam::step() {
    t_++; // Increment the time step
    float bias_corr1 = 1.0f - std::pow(beta1_, t_); // Correct bias for first moment
    float bias_corr2 = 1.0f - std::pow(beta2_, t_); // Correct bias for second moment

    for (const auto& p : params_) {
        Node* n = p.node.get();
        if (!n->requires_grad()) continue;

        // 1. Get gradients and cast to Float32 if necessary
        Tensor grad_f32 = n->grad;
        if (grad_f32.dtype() != Dtype::Float32) {
            grad_f32 = grad_f32.as_type(Dtype::Float32);
        }

        Tensor& m = m_[n];
        Tensor& v = v_[n];

        // 2. Update moments (already in Float32)
        // M  = beta1*M + (1-beta1)*grads;
        m *= beta1_;
        m += (1.0f - beta1_) * grad_f32;

        // V  = beta2*V + (1-beta2)*grads.^2;
        v *= beta2_;
        v += (1.0f - beta2_) * square(grad_f32);

        // 3. Update parameters
        float alpha_eff = alpha_ * std::sqrt(bias_corr2) / bias_corr1;
        Tensor update = alpha_eff * m / (sqrt(v) + epsilon_ * std::sqrt(bias_corr2));

        if (master_params_.count(n)) {
            // Mixed precision path: update master copy and cast back
            Tensor& master_p = master_params_[n];
            master_p -= update;
            n->value.copy_(master_p.as_type(n->value.dtype()));
        } else {
            // Standard path: update directly
            n->value -= update;
        }
    }
}

void Adam::zero_grad() {
    for (const auto& p : params_) {
        Node* n = p.node.get();
        if (n->requires_grad() && n->grad.is_valid()) {
            n->grad = Tensor::zeros(n->grad.shape(), TensorOptions().with_dtype(n->grad.dtype()).with_device(n->grad.device()));
        }
    }
}

const Tensor* Adam::get_master_weight(const Value& v) const { // returns the master weight of the parameter
    auto it = master_params_.find(v.node.get()); // find the master weight of the parameter
    if (it != master_params_.end()) { // if the parameter is found
        return &it->second; // return the master weight
    }
    return nullptr; // if the parameter is not found
}

} // namespace ag