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

class SGDOptimizer {
public:
    SGDOptimizer(const std::vector<Value>& params, float learning_rate = 0.01);
    
    void step();
    void zero_grad();
    const Tensor* get_master_weight(const Value& v) const; // returns the master weight of the parameter

private:
    std::vector<Value> params_;
    float learning_rate_;
    std::unordered_map<Node*, Tensor> master_params_; // Master copy of parameters (Always Float32)
};

class Adam {
public:
    Adam(const std::vector<Value>& params, float alpha = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8);
    
    void step();
    void zero_grad();
    const Tensor* get_master_weight(const Value& v) const; // returns the master weight of the parameter

private:
    std::vector<Value> params_;
    float alpha_;
    float beta1_;
    float beta2_;
    float epsilon_;
    int t_;

    std::unordered_map<Node*, Tensor> m_;             // First moment (Always Float32)
    std::unordered_map<Node*, Tensor> v_;             // Second moment (Always Float32)
    std::unordered_map<Node*, Tensor> master_params_; // Master copy of parameters (Always Float32) stored in the optimizer
};

}