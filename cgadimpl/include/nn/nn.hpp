#pragma once

#include "ad/core/graph.hpp"
#include "ad/ops/ops.hpp"
#include <vector>

namespace ag::nn {

class Module {
public:
    virtual ~Module() = default;
    virtual Value operator()(Value input) = 0; // Takes by value

    Value operator()(const Tensor& input) {
        Value graph_input = ag::make_tensor(input);
        return this->operator()(graph_input);
    }

    std::vector<Value>& parameters() { return params_; }

    void to(Device dev);
    void zero_grad();
    void register_backward_hook(HookFn hook) {
        for (auto& p : params_) {
            p.register_hook(hook);
        }

    }

protected:
    std::vector<Value> params_;
};

class Linear : public Module {
public:
    Linear(int in_features, int out_features, Device dev = Device::CPU);
    Value operator()(Value input) override;
private:
    Value W, b;
};

class Sequential : public Module {
public:
    Sequential(const std::vector<Module*>& modules);
    Value operator()(Value x) override;
    const std::vector<Module*>& get_layers() const { return layers_; }
private:
    std::vector<Module*> layers_;
};

class ReLU : public Module {
public:
    Value operator()(Value input) override;
};

} // namespace ag::nn