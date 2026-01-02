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


class Optimizer {
public:
    Optimizer(const std::vector<Value>& params);
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    void zero_grad();
    const Tensor* get_master_weight(const Value& v) const;

protected:
    std::vector<Value> params_;
    std::unordered_map<Node*, Tensor> master_params_; // Master copy of parameters (Always Float32)
};

class SGDOptimizer : public Optimizer {
public:
    SGDOptimizer(const std::vector<Value>& params, float learning_rate = 0.01);
    
    void step() override;

private:
    float learning_rate_;
};

class Adam : public Optimizer {
public:
    Adam(const std::vector<Value>& params, float alpha = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8);
    
    void step() override;

private:
    float alpha_;
    float beta1_;
    float beta2_;
    float epsilon_;
    int t_;

    std::unordered_map<Node*, Tensor> m_;             // First moment (Always Float32)
    std::unordered_map<Node*, Tensor> v_;             // Second moment (Always Float32)
};

}