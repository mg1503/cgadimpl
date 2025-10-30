// ====================================================================
// FILE: cgadimpl/include/nn/nn.hpp (The Complete Merged Version)
// ====================================================================
#pragma once

#include "ad/graph.hpp" // For Value, Tensor, Device
#include "ad/ops.hpp"   // For matmul, add
#include <vector>

namespace ag::nn {

// --- NEW: High-Level Module API ---

class Module {
public:
    virtual ~Module() = default;
    // Addition a "pure virtual" operator() to the base class.
    // This tells the compiler that all derived classes (like Linear, ReLU)
    // are guaranteed to have a callable forward pass.
    virtual Value operator()(const Value& input) = 0;

    const std::vector<Value>& parameters() const { 
        return params_; 
    }

    void to(Device dev);
    void zero_grad();

protected:
    std::vector<Value> params_;
};

class Linear : public Module {
public:
    Linear(int in_features, int out_features, Device dev = Device::CPU);
    // forward pass declaration
    Value operator()(const Value& input);

private:
    Value W, b;
};

class Sequential : public Module {
public:
    // Takes a list of modules you've created with 'new'
    Sequential(const std::vector<Module*>& modules);
    Value operator()(Value x);

private:
    std::vector<Module*> layers_;
};

class ReLU : public Module {
public:
    Value operator()(const Value& input);
};


} // namespace ag::nn





// // --- OLD: Tensor-based helpers needed by the JIT compiler in graph.cpp ---

// Tensor silu(const Tensor& x);
// Tensor gelu(const Tensor& x);