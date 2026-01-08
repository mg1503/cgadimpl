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

float clip_grad_norm_(std::vector<Value>& params, float max_norm) {
    // Compute total gradient L2 norm across all parameters
    float total_norm_sq = 0.0f;
    
    for (const auto& p : params) {
        Node* n = p.node.get();
        if (!n->requires_grad() || n->grad.numel() == 0) continue;
        
        // Move to CPU for norm computation
        Tensor grad_cpu = n->grad;
        if (grad_cpu.device().device != OwnTensor::Device::CPU) {
            grad_cpu = grad_cpu.to_cpu();
        }
        
        // Compute sum of squares
        const float* g_ptr = grad_cpu.data<float>();
        for (int64_t i = 0; i < grad_cpu.numel(); ++i) {
            total_norm_sq += g_ptr[i] * g_ptr[i];
        }
    }
    
    float total_norm = std::sqrt(total_norm_sq);
    
    // Clip gradients if norm exceeds max_norm
    if (total_norm > max_norm) {
        float clip_coef = max_norm / (total_norm + 1e-6f);
        for (const auto& p : params) {
            Node* n = p.node.get();
            if (!n->requires_grad() || n->grad.numel() == 0) continue;
            n->grad *= clip_coef;
        }
    }
    
    return total_norm;
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
        
        // Skip if gradient is empty or has wrong shape
        if (n->grad.numel() == 0) continue;
        if (n->grad.shape().dims != n->value.shape().dims) continue;

        Tensor grad = n->grad;
        Tensor& m = m_[n];
        Tensor& v = v_[n];
        
        // Ensure gradient is on same device as parameter
        if (grad.device().device != n->value.device().device) {
            if (n->value.device().device == OwnTensor::Device::CUDA) {
                grad = grad.to(n->value.device());
            } else {
                grad = grad.to_cpu();
            }
        }

        // M  = beta1*M + (1-beta1)*grads;
        // Using out-of-place operations to avoid broadcast issues
        m = m * beta1_ + (1.0f - beta1_) * grad;

        // V  = beta2*V + (1-beta2)*grads.^2;
        v = v * beta2_ + (1.0f - beta2_) * square(grad);

        float alpha_eff = alpha_ * std::sqrt(bias_corr2) / bias_corr1;
        
        // params = params - alpha_eff * m / (sqrt(v) + epsilon)
        n->value = n->value - alpha_eff * m / (sqrt(v) + epsilon_ * std::sqrt(bias_corr2));
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