// // =========================================================
// // FILE: cgadimpl/include/nn/nn.hpp
// // =========================================================
// #pragma once

// #include "ad/graph.hpp" // For Value, Tensor, Device
// #include "ad/ops.hpp"   // For matmul, add
// #include <vector>

// namespace ag::nn {

// /**
//  * @brief Base class for all neural network modules.
//  * 
//  * Manages parameters and provides utilities for moving them between devices.
//  */
// class Module {
// public:
//     virtual ~Module() = default;

//     /**
//      * @brief Returns a vector containing all learnable parameters of the module.
//      */
//     const std::vector<Value>& parameters() const {
//         return params_;
//     }

//     /**
//      * @brief Moves all module parameters to the specified device (CPU or CUDA).
//      */
//     void to(Device dev);

//     /**
//      * @brief Zeros out the gradients of all parameters.
//      */
//     void zero_grad();

// protected:
//     // Vector to store learnable parameters. Derived classes should populate this.
//     std::vector<Value> params_;
// };

// /**
//  * @brief A fully connected linear layer: y = xA^T + b.
//  * 
//  * Note: We store weights as (in_features, out_features) and use matmul(x, W).
//  */
// class Linear : public Module {
// public:
//     /**
//      * @param in_features   Size of each input sample.
//      * @param out_features  Size of each output sample.
//      * @param dev           The device to create the parameters on.
//      */
//     Linear(int in_features, int out_features, Device dev = Device::CPU);

//     /**
//      * @brief Performs the forward pass of the linear layer.
//      */
//     Value operator()(const Value& input);

// private:
//     Value W, b;
// };

// } // namespace ag::nn

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
    const std::vector<Value>& parameters() const { return params_; }
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

// --- OLD: Tensor-based helpers needed by the JIT compiler in graph.cpp ---

Tensor silu(const Tensor& x);
Tensor gelu(const Tensor& x);
} // namespace ag::nn