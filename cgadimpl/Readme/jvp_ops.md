# JVP Ops Documentation

This document provides a detailed overview of the Jacobian-Vector Product (JVP) operations within the `cgadimpl` codebase. These functions are responsible for the **forward-mode** automatic differentiation (computing "tangents" or directional derivatives). The implementation is located in `autodiff_jvp_ops.cpp`.

## Overview

The `autodiff_jvp_ops` module implements the forward-mode differentiation rules. For every operation in the graph (e.g., `z = f(x, y)`), there is a corresponding `jvp_*` function that calculates the tangent of the output `t_z` given the tangents of the inputs `t_x` and `t_y` using the chain rule:
`t_z = (df/dx * t_x) + (df/dy * t_y)`

## Namespaces Used

*   **`ag`**: The primary namespace.
*   **`ag::detail`**: Internal namespace where all `jvp_*` functions are defined.

## Dependencies

The module relies on:

| File / Component | Type | Description |
| :--- | :--- | :--- |
| `ad/detail/autodiff_ops.hpp` | Header | Declares the JVP function signatures (`JvpFn`) and lookup table. |
| `TensorLib.h` (via `OwnTensor`) | External | Provides the *stream-aware* math operations (`OwnTensor::matmul`, `OwnTensor::reduce_sum`) used to compute tangents. |
| `ad/core/graph.hpp` | Internal | Access to `Node` structure and `Node::value` (the primal value). |
| `ad/runtime/runtime.hpp` | Header | Provides access to `ag::current_stream()` for CUDA operations. |

## Functions Declared

The core interface is the `jvp_lookup(Op op)` function, which dispatches to specific implementations:

```cpp
// JVP Signature
// t(n) is a callback that returns the tangent of the input node n
using JvpFn = Tensor(*)(Node* n, const std::function<const Tensor&(Node*)>& t);
```

### Functionality Breakdown

| Function | Functionality | Dependencies |
| :--- | :--- | :--- |
| **`jvp_Add`, `jvp_Sub`** | Linearity of differentiation: `t(A) + t(B)`. | `Tensor::operator+` |
| **`jvp_Mul`** | Product rule: `t(A)*B + A*t(B)`. | `Tensor::operator*` |
| **`jvp_MatMul`** | Product rule for matrices: `t(A)@B + A@t(B)`. | `OwnTensor::matmul` |
| **`jvp_Relu`** | Forward differentiation of ReLU. Computes mask (`value > 0`) and applies it to tangent: `t(X) * mask`. | `OwnTensor::abs` |
| **`jvp_Exp`, `jvp_Log`** | `t(x) * exp(x)`, `t(x) / x`. | `Tensor` ops |
| **`jvp_Tanh`, `jvp_Sigmoid`** | Derivative dependent on output: `t(x) * (1 - tanh^2)`, `t(x) * s * (1-s)`. | `Tensor` ops |
| **`jvp_SoftmaxRow`** | Tangent of softmax: `y * (t(z) - dot(y, t(z)))`. | `OwnTensor::reduce_sum` |
| **`jvp_MSELoss`** | Dot product of gradients with tangents: `(dL/dZ . tZ) + (dL/dY . tY)`. | `OwnTensor` |
| **`jvp_CeWithLogits`** | Computes `dot(gZ, tZ) + dot(gY, tY)` where `gZ` and `gY` are gradients derived from stable softmax. | `OwnTensor` |
| **`jvp_RMSNorm`** | Computes tangent for RMS Norm. Includes complex terms for derivative of `rsqrt(variance)`. | `OwnTensor` |

## Analysis: Pros and Cons

### Pros
*   **Device Agnostic**: Like VJP, it uses `OwnTensor` arithmetic which is stream-aware and works on CPU/GPU.
*   **Correctness**: Basic arithmetic and activation functions (`SiLU`, `GELU`, `LeakyRelu`) seem correctly implemented with their respective derivatives.
*   **Loss Function Support**: Implements full JVPs for `MSELoss` and `MAELoss` (via arithmetic sign approximation), allowing forward-mode checking of loss layers.

### Cons, Drawbacks, and Issues
*   **Missing Implementations**: **Major Drawback**. Compared to the VJP module, JVP support is significantly incomplete. The following operations throw `std::runtime_error`:
    *   **Attention Family**: `Attention`, `AlibiAttention`, `RELUAtt`, `SigAtt`.
    *   **Normalization**: `LayerNorm`, `RealLayerNorm`, `RealRMSNorm` (only basic `RMSNorm` is implemented).
    *   **Advanced Ops**: `SWIGLU`, `MOE`, `Dyntanh`, `KLDivergence`.
*   **Missing Argmax**: `jvp_RowMax` is unimplemented because `OwnTensor` lacks an `argmax` or comparison operator.
*   **API Complexity**: The JVP function signature requires passing a callback `t` (tangent accessor), which is slightly more complex to use/test in isolation than the VJP's explicit gradient passing.

## Implementation Status Table

| Category | Fully Implemented | Missing / Unimplemented |
| :--- | :--- | :--- |
| **Arithmetic** | Add, Sub, Mul, Div, FMA, Linear, Reciprocal | - |
| **activation** | Relu, Sigmoid, Tanh, Gelu, LeakyRelu, Softplus, Silu, Mish, Gaus, Parcon, Lisht, Sign | - |
| **Loss** | MSELoss, MAELoss, CeWithLogits | **KLDivergence** |
| **Normalization** | RMSNorm | **LayerNorm, RealLayerNorm, RealRMSNorm** |
| **Graph / Shape** | MatMul, Transpose, Sum, RowSum, MeanAll, SoftmaxRow, LogSumExpRow | **RowMax, MOE** |
| **Attention** | - | **Attention, Alibi, SigAtt, RELUAtt** |
| **Advanced** | - | **SWIGLU, Dyntanh** |
