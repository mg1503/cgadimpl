# Optimizer Documentation

This document provides a detailed overview of the Optimizer subsystem within the `cgadimpl` codebase. It outlines the namespaces, functions, functionality, dependencies, and an analysis of the pros and cons of the current implementation.

## Overview

The `optim` module provides algorithms to update the parameters of a model to minimize a loss function. Currently, it implements a basic **Stochastic Gradient Descent (SGD)** optimizer. Unlike PyTorch, where optimizers are classes that hold references to parameters, this implementation is a functional pass over the computation graph.

## Namespaces Used

*   **`ag`**: The primary namespace for the Autodiff Graph library.

## Dependencies

The `optim` module relies on the following internal components:

| File / Component | Type | Description |
| :--- | :--- | :--- |
| `ad/core/graph.hpp` | Internal | Access to `Node` and `Value` to traverse the graph and identify parameters (`Op::Leaf`). |
| `tensor.hpp` | Internal | Access to `Tensor` operator overloads (`*`, `+=`) to perform the weight update arithmetic on the correct device. |

## Functions Declared

The following function is declared in `include/ag/optim.hpp`:

```cpp
namespace ag {

    void SGD(const Value& root, const Tensor* grad_seed = nullptr, float learning_rate = 100);

}
```

## Functionality Breakdown

| Function | Functionality | Dependencies |
| :--- | :--- | :--- |
| **`SGD`** | Performs a gradient descent step. <br> 1. **Traversal**: Topologically sorts the graph starting from `root`. <br> 2. **Identification**: Iterates through all nodes to find those that are **Leaves** (`Op::Leaf`) and **Require Gradient** (`requires_grad()`). These are the trainable parameters. <br> 3. **Update**: Modifies the parameter value in-place: `param -= lr * grad`. | `topo_from`, `Node::is_leaf`, `Tensor::operator+=` |

## Analysis: Pros and Cons

### Pros
*   **Simplicity**: The implementation is extremely minimal (less than 30 lines of code). It doesn't require maintaining a separate list of parameters; it just finds them in the graph.
*   **Device Agnostic**: By using the `Tensor` operator overloads, the update logic works automatically on CPU or GPU without custom kernels.

### Cons & Potential Drawbacks
*   **Inefficient Traversal**: It traverses the *entire* compute graph (which could be thousands of nodes) just to find the handful of leaf parameters at the bottom. A standard optimizer holds a pre-compiled list of `std::vector<Tensor*> params` to iterate over, which is O(P) instead of O(N).
*   **Limited Features**:
    *   No **Momentum** or **Velocity** tracking (cannot implement Adam/SGD+Momentum easily because it relies on stateless graph traversal).
    *   No **Weight Decay**.
    *   No **Gradient Clipping**.
*   **Graph Dependency**: It requires the full graph to be alive. You cannot delete the graph and keep the parameters easily, whereas PyTorch optimizers own the parameters independently of the compute graph.
*   **No "Zero Grad"**: It assumes gradients are populated. It does not offer a `step()` vs `zero_grad()` distinction directly (though `autodiff::zero_grad` exists separately).
