# VJP Ops Documentation

This document provides a detailed overview of the Vector-Jacobian Product (VJP) operations within the `cgadimpl` codebase. These functions are responsible for the **backward pass** of automatic differentiation, calculating gradients for each operation in the graph. The implementation is located in `autodiff_vjp_ops.cpp`.

## Overview

The `autodiff_vjp_ops` module implements the reverse-mode differentiation rules. For every forward operation (like `Add`, `MatMul`, `Attention`), there is a corresponding `vjp_*` function that:
1.  Receives the upstream gradient `gy` (tensor of gradients with respect to the output).
2.  Computes the downstream gradients for each input.
3.  Accumulates these gradients into the inputs' `grad` tensors (`input->grad += ...`).

## Namespaces Used

*   **`ag`**: The primary namespace.
*   **`ag::detail`**: Internal namespace where all `vjp_*` functions are defined.

## Dependencies

The module relies heavily on:

| File / Component | Type | Description |
| :--- | :--- | :--- |
| `ad/detail/autodiff_ops.hpp` | Header | Declares the VJP function signatures and lookup table. |
| `TensorLib.h` (via `OwnTensor`) | External | Provides the *stream-aware* math operations (`OwnTensor::matmul`, `OwnTensor::reduce_sum`, `OwnTensor::transpose`) used to compute gradients on CPU or GPU. |
| `ad/core/graph.hpp` | Internal | Access to `Node`, `Tensor`, and the `tape` (for retrieving saved intermediate values). |

## Functions Declared

The core interface is the `vjp_lookup(Op op)` function, which dispatches to specific implementations:

```cpp
// VJP Signature
using VjpFn = void(*)(Node* n, const Tensor& gy);
```

### Functionality Breakdown

| Function | Functionality | Dependencies |
| :--- | :--- | :--- |
| **`vjp_Add`, `vjp_Sub`** | Computes `gy` for inputs. Handles broadcasting via `reduce_for_broadcast` (internal helper). | `reduce_for_broadcast` |
| **`vjp_Mul`, `vjp_Div`** | Standard product/quotient rules. `dA = gy * B`, `dB = gy * A`. | `Tensor` ops |
| **`vjp_MatMul`** | Matrix multiplication gradient: `dA = dY @ B.T`, `dB = A.T @ dY`. | `OwnTensor::matmul` |
| **`vjp_LayerNorm`** | Complex VJP. Recomputes standard deviation from taped `variance`. Implements exact gradient for `(x-u)/sigma`. | `OwnTensor::reduce_sum` |
| **`vjp_Attention`** | **Critical**. Retrieves `q, k, v, s` from `n->tape`. Backprops through Softmax, then through Q, K, V projections. Updates gradients for 3 weight matrices and input A. | `OwnTensor` |
| **`vjp_Relu`** | **Stable Implementation**. Recreates the mask using the *output* value (`n->value > 0`), which is numerically safer than using input `x`. | `OwnTensor::abs` |
| **`vjp_GELU`** | Approximated GELU derivative involves `tanh` and polynomials. | `OwnTensor::tanh` |
| **`vjp_MSELoss`** | Mean Squared Error gradient: `2/N * (pred - target)`. | Arithmetic |
| **`vjp_CeWithLogits`** | Cross Entropy gradient: `(softmax(z) - y) / N`. Recomputes stable softmax. | `OwnTensor::exp` |
| **`vjp_SWIGLU`** | Recomputes the entire forward pass (SiLU gate) to calculate gradients for 3 different linear projections. | `OwnTensor` |

## Analysis: Pros and Cons

### Pros
*   **Device Agnostic**: The active code uses `OwnTensor::*` methods (like `OwnTensor::abs`, `OwnTensor::matmul`), which automatically handle CUDA streams and device dispatch.
*   **Tape Efficiency**: Complex ops like `Attention` and `RMSNorm` correctly use the `tape` to retrieve intermediate results (variance, softmax output) instead of recomputing them, which is a key optimization.
*   **Stability**: The `vjp_Relu` implementation explicitly handles numerical stability by using the output sign. `CeWithLogits` implements stable `log_softmax` for gradients.

### Cons, Drawbacks, and Issues
*   **Dead Code**: **Major Con**. The first **1,061 lines** of `autodiff_vjp_ops.cpp` are completely commented out. This version appears to be a legacy CPU-only implementation. The actual code starts at line 1062. This significantly bloats the file size and confuses maintenance.
*   **Missing Implementations**:
    *   `vjp_RealRMSNorm`: Throws `std::runtime_error` ("not implemented yet").
    *   `vjp_RowMax`: Throws `std::runtime_error` because `argmax` is missing from the underlying tensor library.
*   **Broadcasting Fragility**: `vjp_Sum` and `vjp_RowSum` rely on implicit `+=` broadcasting. While concise, explicit reshapes are often safer in C++ tensor libraries to prevent accidental mismatches.
*   **Implementation Gaps**: `vjp_MAELoss` has to manually implement `sign` using `abs` logic (`x / abs(x)`), suggesting `OwnTensor` lacks a basic `sign()` operator for gradients.

## List of Operations

| Category | Operations Implemented | Missing / Broken |
| :--- | :--- | :--- |
| **Arithmetic** | Add, Sub, Mul, Div, FMA, Reciprocal | - |
| **Activation** | Relu, Sigmoid, Tanh, Gelu, LeakyRelu, Softplus, Silu, Mish, Gaus, Parcon, Lisht, Sign | - |
| **Loss** | MSELoss, MAELoss, CeWithLogits, KLDivergence | - |
| **Normalization** | LayerNorm, RMSNorm, RealLayerNorm | **RealRMSNorm** (Not Implemented) |
| **Trig** | Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh | - |
| **Graph / Shape** | MatMul, Transpose, MOE (Mixture of Experts) | **RowMax** (No argmax) |
| **Attention** | Attention, AlibiAttention, SigAtt, RELUAtt | - |
