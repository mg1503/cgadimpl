// =====================
// file: cgadimpl/include/ag/optim.hpp (declarations only)
// =====================
#pragma once
#include "ad/ops/ops.hpp"
#include "ad/utils/debug.hpp"
#include "ad/autodiff/autodiff.hpp"
#include "ad/detail/autodiff_ops.hpp"
#include "tensor.hpp"
#include "ad/utils/debug.hpp"
#include <math.h>

#include <unordered_map>
#include <vector>

namespace ag {

void SGD(const Value& root, const Tensor* grad_seed=nullptr, float learning_rate=100);

// Clips gradient norm of parameters in-place
// Returns the total norm of the gradients before clipping
float clip_grad_norm_(std::vector<Value>& params, float max_norm);

class Adam {
public:
    Adam(const std::vector<Value>& params, float alpha = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8);
    
    void step();
    void zero_grad();
    void set_alpha(float alpha) { alpha_ = alpha; }
    float get_alpha() const { return alpha_; }

private:
    std::vector<Value> params_;
    float alpha_;
    float beta1_;
    float beta2_;
    float epsilon_;
    int t_;

    std::unordered_map<Node*, Tensor> m_; // First moment
    std::unordered_map<Node*, Tensor> v_; // Second moment
};

}