// ===================================================================
// file: cgadimpl/src/optim.cpp (Corrected for OwnTensor)
// ===================================================================
#include "optim.hpp"
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

Adam::Adam(const std::vector<Value>& params, float alpha, float beta1, float beta2, float epsilon)
    : params_(params), alpha_(alpha), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {
    
    for (const auto& p : params_) {
        Node* n = p.node.get();
        if (n->requires_grad()) {
            // Initialize moments with zeros on the same device as the parameter
            m_[n] = Tensor::zeros(n->value.shape(), options(n->value));
            v_[n] = Tensor::zeros(n->value.shape(), options(n->value));
        }
    }
}

void Adam::step() {
    t_++;
    float bias_corr1 = 1.0f - std::pow(beta1_, t_);
    float bias_corr2 = 1.0f - std::pow(beta2_, t_);

    for (const auto& p : params_) {
        Node* n = p.node.get();
        if (!n->requires_grad()) continue;

        Tensor& grad = n->grad;
        Tensor& m = m_[n];
        Tensor& v = v_[n];

        // M  = beta1*M + (1-beta1)*grads;
        // Using in-place to avoid unnecessary allocations
        m *= beta1_;
        m += (1.0f - beta1_) * grad;

        // V  = beta2*V + (1-beta2)*grads.^2;
        v *= beta2_;
        v += (1.0f - beta2_) * square(grad);

        // M2 = M / (1-beta1^iT);
        // V2 = V / (1-beta2^iT);
        // alpha_eff = alpha * sqrt(1-beta2^iT)/(1-beta1^iT);
        // params = params - alpha_eff * m / (sqrt(v) + epsilon);
        
        // Note: The MATLAB implementation is equivalent to the standard bias-corrected update:
        // params = params - alpha * (m / bias_corr1) / (sqrt(v / bias_corr2) + epsilon)
        
        float alpha_eff = alpha_ * std::sqrt(bias_corr2) / bias_corr1;
        
        // We add epsilon inside the sqrt context matching the MATLAB logic: sqrt(V2) + epsilon
        // Since V2 = V / bias_corr2, sqrt(V2) = sqrt(V) / sqrt(bias_corr2)
        // Thus: m_hat / (sqrt(v_hat) + epsilon) = (m/bias_corr1) / (sqrt(v)/sqrt(bias_corr2) + epsilon)
        // = (m * sqrt(bias_corr2) / bias_corr1) / (sqrt(v) + epsilon * sqrt(bias_corr2))
        
        n->value -= alpha_eff * m / (sqrt(v) + epsilon_ * std::sqrt(bias_corr2));
    }
}

void Adam::zero_grad() {
    for (const auto& p : params_) {
        Node* n = p.node.get();
        if (n->requires_grad()) {
            n->grad.fill(0.0f);
        }
    }
}

} // namespace ag