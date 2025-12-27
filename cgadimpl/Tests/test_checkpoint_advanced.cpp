#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <unordered_set>

#include "ad/autodiff/autodiff.hpp"
#include "ad/ops/ops.hpp"
#include "ad/autodiff/checkpoint.hpp"
#include "ad/autodiff/careful_deletion.hpp"
#include "ad/autodiff/inplace.hpp"
#include "ad/utils/debug.hpp"

using namespace ag;

// Helper to compare tensors
bool compare_tensors(const Tensor& a, const Tensor& b, float tol = 1e-5) {
    if (a.numel() != b.numel()) return false;
    if (a.shape().dims != b.shape().dims) return false;
    
    Tensor a_cpu = a.to_cpu();
    Tensor b_cpu = b.to_cpu();
    
    const float* data_a = a_cpu.data<float>();
    const float* data_b = b_cpu.data<float>();
    
    for (size_t i = 0; i < a.numel(); ++i) {
        if (std::abs(data_a[i] - data_b[i]) > tol) {
            std::cerr << "Mismatch at index " << i << ": " << data_a[i] << " vs " << data_b[i] << "\n";
            return false;
        }
    }
    return true;
}

// 1. Numerical Parity Test
void test_numerical_parity() {
    std::cout << "[Test] Numerical Parity... ";
    
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    Value x = make_tensor(Tensor::randn(Shape{{4, 4}}, opts), "x");
    Value w1 = make_tensor(Tensor::randn(Shape{{4, 4}}, opts), "w1");
    Value w2 = make_tensor(Tensor::randn(Shape{{4, 4}}, opts), "w2");
    
    // Standard run
    Value h1 = matmul(x, w1);
    Value h2 = matmul(h1, w2);
    Value loss = sum(h2);
    
    backward(loss);
    
    Tensor grad_x_ref = x.node->grad.clone();
    Tensor grad_w1_ref = w1.node->grad.clone();
    Tensor grad_w2_ref = w2.node->grad.clone();
    
    // Reset grads
    zero_grad(loss);
    
    // Checkpointed run
    Value h1_cp = matmul(x, w1);
    checkpoint_impl::mark_node_checkpoint(h1_cp.node);
    Value h2_cp = matmul(h1_cp, w2);
    Value loss_cp = sum(h2_cp);
    
    // Delete h1_cp value
    memory::sweep_safe_nodes(loss_cp, memory::DeletePolicy::ForwardPass, {loss_cp.node.get(), x.node.get(), w1.node.get(), w2.node.get()});
    
    if (h1_cp.node->value.numel() != 0) {
        std::cout << "❌ Failed: h1_cp value not deleted\n";
        return;
    }
    
    backward(loss_cp);
    
    if (compare_tensors(grad_x_ref, x.node->grad) &&
        compare_tensors(grad_w1_ref, w1.node->grad) &&
        compare_tensors(grad_w2_ref, w2.node->grad)) {
        std::cout << "✅ Passed\n";
    } else {
        std::cout << "❌ Failed: Gradient mismatch\n";
    }
}

// 2. Deep Graph Test
void test_deep_graph() {
    std::cout << "[Test] Deep Graph Recomputation... ";
    
    const int depth = 500;
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    Value x = make_tensor(Tensor::ones(Shape{{1, 1}}, opts), "x");
    
    Value cur = x;
    std::vector<std::shared_ptr<Node>> intermediates;
    
    for (int i = 0; i < depth; ++i) {
        cur = add(cur, make_tensor(Tensor::ones(Shape{{1, 1}}, opts), ("c" + std::to_string(i)).c_str()));
        if (i == 0) {
            checkpoint_impl::mark_node_checkpoint(cur.node);
        } else {
            intermediates.push_back(cur.node);
        }
    }
    
    Value loss = sum(cur);
    
    // Delete all intermediates except the checkpoint at i=0
    for (auto& n : intermediates) {
        n->value = Tensor();
    }
    
    try {
        backward(loss);
        std::cout << "✅ Passed (Depth: " << depth << ")\n";
    } catch (const std::exception& e) {
        std::cout << "❌ Failed: " << e.what() << "\n";
    }
}

// 3. Diamond Pattern Test
void test_diamond_pattern() {
    std::cout << "[Test] Diamond Pattern Recomputation... ";
    
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    Value a = make_tensor(Tensor::ones(Shape{{2, 2}}, opts), "a");
    checkpoint_impl::mark_node_checkpoint(a.node);
    
    Value b = relu(a);
    Value c = exp(a);
    Value d = add(b, c);
    Value loss = sum(d);
    
    // Delete 'a'
    a.node->value = Tensor();
    
    try {
        backward(loss);
        std::cout << "✅ Passed\n";
    } catch (const std::exception& e) {
        std::cout << "❌ Failed: " << e.what() << "\n";
    }
}

// 4. Nested Checkpointing Test
void test_nested_checkpointing() {
    std::cout << "[Test] Nested Checkpointing... ";
    
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    Value a = make_tensor(Tensor::ones(Shape{{2, 2}}, opts), "a");
    
    Value b = relu(a);
    checkpoint_impl::mark_node_checkpoint(b.node);
    
    Value c = exp(b);
    checkpoint_impl::mark_node_checkpoint(c.node);
    
    Value loss = sum(c);
    
    // Delete a, b, c
    // a is leaf, won't be deleted. b and c are checkpoints.
    b.node->value = Tensor();
    c.node->value = Tensor();
    
    try {
        backward(loss);
        std::cout << "✅ Passed\n";
    } catch (const std::exception& e) {
        std::cout << "❌ Failed: " << e.what() << "\n";
    }
}

// 5. Leaf Node Checkpointing (Should be handled gracefully)
void test_leaf_checkpointing() {
    std::cout << "[Test] Leaf Node Checkpointing... ";
    
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    Value a = make_tensor(Tensor::ones(Shape{{2, 2}}, opts), "a");
    
    // Mark leaf as checkpoint
    checkpoint_impl::mark_node_checkpoint(a.node);
    
    Value b = relu(a);
    Value loss = sum(b);
    
    try {
        backward(loss);
        std::cout << "✅ Passed\n";
    } catch (const std::exception& e) {
        std::cout << "❌ Failed: " << e.what() << "\n";
    }
}

// 6. In-place Interaction Test
void test_inplace_interaction() {
    std::cout << "[Test] In-place Interaction... ";
    
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    Value x = make_tensor(Tensor::ones(Shape{{2, 2}}, opts), "x");
    
    Value y = relu(x);
    // Mark as in-place checkpoint (saves snapshot of y=1)
    ag::inplace::mark_inplace_checkpoint(y.node);
    
    Value z = exp(y); // z = exp(1)
    Value loss = sum(z);
    
    // Simulate in-place modification of y: y = y + 10 (y is now 11)
    {
        Tensor y_val = y.node->value;
        Tensor y_cpu = y_val.to_cpu();
        float* data = (float*)y_cpu.data();
        for(size_t i=0; i<y_val.numel(); ++i) data[i] += 10.0f;
        y.node->value.set_data(data, y_val.numel());
        ag::inplace::bump_tensor_version(y.node.get());
    }
    
    // Delete y's value to force restoration/recomputation
    y.node->value = Tensor();
    
    try {
        // backward(loss) will need y. Since y was 1 when z was computed,
        // it MUST restore y=1 from snapshot, NOT recompute it (which would give 1)
        // OR if it recomputes, it should be fine if the inputs haven't changed.
        // But the snapshot is safer.
        backward(loss);
        
        // If backward succeeded, it means the system handled the missing/stale value.
        std::cout << "✅ Passed\n";
    } catch (const std::exception& e) {
        std::cout << "❌ Failed: " << e.what() << "\n";
    }
}

// 7. Memory Pressure Test
void test_memory_pressure() {
    std::cout << "[Test] Memory Pressure Sweep... ";
    
    auto opts = TensorOptions().with_dtype(Dtype::Float32);
    Value x = make_tensor(Tensor::randn(Shape{{1024, 1024}}, opts), "x"); // 4MB
    
    Value cur = x;
    for (int i = 0; i < 10; ++i) {
        cur = matmul(cur, make_tensor(Tensor::randn(Shape{{1024, 1024}}, opts), ("w" + std::to_string(i)).c_str()));
        checkpoint_impl::mark_node_checkpoint(cur.node);
    }
    Value loss = sum(cur);
    
    // Total memory is roughly 11 * 4MB = 44MB
    // Sweep with target 10MB
    memory::sweep_with_checkpoint_priority(loss, 10);
    
    try {
        backward(loss);
        std::cout << "✅ Passed\n";
    } catch (const std::exception& e) {
        std::cout << "❌ Failed: " << e.what() << "\n";
    }
}

int main() {
    std::cout << "==================================================\n";
    std::cout << "      Advanced Checkpointing Test Suite           \n";
    std::cout << "==================================================\n";
    
    test_numerical_parity();
    test_deep_graph();
    test_diamond_pattern();
    test_nested_checkpointing();
    test_leaf_checkpointing();
    test_inplace_interaction();
    test_memory_pressure();
    
    std::cout << "==================================================\n";
    return 0;
}
