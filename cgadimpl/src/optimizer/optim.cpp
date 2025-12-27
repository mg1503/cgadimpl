// ===================================================================
// file: cgadimpl/src/optim.cpp (Corrected for OwnTensor)
// ===================================================================
#include "optim.hpp"
#include <math.h>

// No new includes are needed because tensor.hpp brings in everything.

namespace ag {

void SGD(const Value& root, const Tensor* grad_seed, float learning_rate) { // Changed to float for consistency
    auto order = topo_from(root.node.get());

    // NOTE: The 'backward' function is responsible for seeding the initial gradient.
    // The SGD optimizer's job is just to update the weights.
    // Therefore, the "seed" block is not actually needed here, as backward() must
    // have been called first, which would have already populated all the gradients.
    // We can remove it for a cleaner implementation.

    // The loop now correctly iterates forward to find Leaf nodes that are parameters.
    for (Node* n : order) {
        // We only update nodes that are trainable parameters.
        // In our design, these are Leaf nodes that require a gradient.
        if (n->op == Op::Leaf && n->requires_grad()) {
            
            // --- THE FIX: Use the new, stream-aware operators ---
            
            // 1. (-learning_rate * n->tensor.grad_view())
            // This calls the overloaded operator*(float, Tensor), which correctly
            // gets the stream from the context for GPU operations.
            
            // 2. n->tensor += ...
            // This calls the overloaded operator+=, which also correctly
            // gets the stream from the context for GPU operations.
            n->tensor += -learning_rate * n->tensor.grad_view();
        }
    }
}

} // namespace ag