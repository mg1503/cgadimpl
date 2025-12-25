

#include <iostream>
#include <cmath>
#include "ad/ag_all.hpp"
#include "ad/autodiff/inplace.hpp"
#include "ad/autodiff/careful_deletion.hpp"
#include <iostream>
#include <cmath>

// Main framework headers
#include "ad/ag_all.hpp"
#include "ad/autodiff/inplace.hpp"
#include "ad/autodiff/careful_deletion.hpp"

using namespace ag;

// Helper function to compare tensors using the correct data access API
bool allclose(const Tensor& A, const Tensor& B, float tol = 0.5f) {
    if (A.shape().dims != B.shape().dims) return false;
    
    // Tensors must be on CPU to access data directly
    Tensor A_cpu = A.to_cpu();
    Tensor B_cpu = B.to_cpu();
    
    const float* a_data = A_cpu.data<float>();
    const float* b_data = B_cpu.data<float>();
    
    for (size_t i = 0; i < A.numel(); ++i) {
        if (std::fabs(a_data[i] - b_data[i]) > tol) {
            std::cerr << "Mismatch at index " << i << ": " << a_data[i] << " vs " << b_data[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    std::cout << "===== COMPLEX CAREFUL DELETION TEST =====\n";

    // -------------------------------------------------------------
    // 1️⃣ BUILD A COMPLEX COMPUTATION GRAPH
    // -------------------------------------------------------------
    auto opts_param = TensorOptions().with_req_grad(true);
    Tensor Ta = Tensor::randn(Shape{{3, 3}}, opts_param);
    Tensor Tb = Tensor::randn(Shape{{3, 3}}, opts_param);
    Tensor Tc = Tensor::randn(Shape{{3, 3}}, opts_param);
    Tensor Td = Tensor::randn(Shape{{3, 3}}, opts_param);

    Value a = make_tensor(Ta, "a");
    Value b = make_tensor(Tb, "b");
    Value c = make_tensor(Tc, "c");
    Value d = make_tensor(Td, "d");

    // Layer 1
   // --- FIX: Checkpoint the parent node as well ---
    
    // Layer 1
    Value l1_parent = add(matmul(a, b), c);
    l1_parent = checkpoint(l1_parent, CheckpointOptions{}); // MARK THE PARENT
    
    Value l1 = relu(l1_parent);
    l1 = checkpoint(l1, CheckpointOptions{}); // MARK THE CHILD
    
    // --- END FIX ---

    // Layer 2 - creates an alias intentionally
    Value l2 = l1; // alias to l1
    // Explicitly register alias to test the inplace subsystem
    ag::inplace::register_tensor_alias((void*)l1.val().data(), l1.node.get());
    ag::inplace::register_tensor_alias((void*)l2.val().data(), l2.node.get());

    // Layer 3
    Value l3 = add(matmul(l2, d), c);
    l3 = inplace_checkpoint(l3); // Mark for in-place checkpointing
    // Note: inplace_checkpoint() already calls mark_inplace_checkpoint internally.
    
    // Layer 4 - nonlinearity
    Value l4 = relu(add(l3, c));

    // Output layer: sum of all elements
    Value loss = sum(l4);

    std::cout << "[Graph Build] Completed forward graph construction.\n";

    // -------------------------------------------------------------
    // 2️⃣ FORWARD AND BACKWARD
    // -------------------------------------------------------------
    backward(loss);
    std::cout << "\n[Backward] Gradients computed successfully.\n";

    debug::print_grad("a.grad", a);
    debug::print_grad("b.grad", b);
    debug::print_grad("c.grad", c);
    debug::print_grad("d.grad", d);

    // -------------------------------------------------------------
    // 3️⃣ VERSION TRACKING AFTER FORWARD
    // -------------------------------------------------------------
    size_t v1 = inplace::get_tensor_version(l1.node.get());
    size_t v3 = inplace::get_tensor_version(l3.node.get());
    std::cout << "\n[Version Tracking]\n";
    std::cout << "  Layer1 (checkpointed) version = " << v1 << "\n";
    std::cout << "  Layer3 (inplace) version = " << v3 << "\n";

    // Perform in-place modification to test version bump
    std::cout << "\n[In-place Modification Test]\n";
    l3.node->value += (Tensor::ones(l3.val().shape(), ag::options(l3.val())) * 0.1f);
    inplace::bump_tensor_version(l3.node.get());
    size_t v3_after = inplace::get_tensor_version(l3.node.get());
    std::cout << "  Layer3 version after in-place update = " << v3_after << "\n";

    // -------------------------------------------------------------
    // 4️⃣ RUN CAREFUL DELETION (SAFE MODE)
    // -------------------------------------------------------------
    std::cout << "\n[Careful Deletion: SAFE MODE]\n";
    ag::memory::sweep_safe_nodes(loss, ag::memory::DeletePolicy::AlwaysSafe);
    std::cout << "  ✔ Safe deletion completed.\n";
    std::cout << "  Check if checkpoint node is still protected: "
              << (l1.node->is_checkpoint ? "✅ yes\n" : "❌ no\n");

    // -------------------------------------------------------------
    // 5️⃣ SIMULATE VALUE DROPS AND RECOMPUTE
    // -------------------------------------------------------------
    std::cout << "\n[Simulate Deallocation + Recomputation]\n";
    // Correctly create empty tensors to simulate deallocation
    l1.node->value = Tensor(Shape{}, TensorOptions{});
    l3.node->value = Tensor(Shape{}, TensorOptions{});

    bool ok1 = ag::checkpoint_impl::recompute_subgraph(l1.node);
    // For in-place, the correct recompute function is different
    bool ok3 = ag::inplace::recompute_inplace(l3.node); 
    std::cout << "  Layer1 recompute: " << (ok1 ? "✅ success\n" : "❌ fail\n");
    std::cout << "  Layer3 recompute: " << (ok3 ? "✅ success\n" : "❌ fail\n");

    if (ok1) {
        debug::print_value("l1 (recomputed)", l1);
    }
    if (ok3) {
        debug::print_value("l3 (recomputed)", l3);
    }

    // -------------------------------------------------------------
    // 6️⃣ CAREFUL DELETION (AGGRESSIVE MODE)
    // -------------------------------------------------------------
    std::cout << "\n[Careful Deletion: AGGRESSIVE MODE]\n";
    ag::memory::sweep_safe_nodes(loss, ag::memory::DeletePolicy::Aggressive);

    bool ok1b = ag::checkpoint_impl::recompute_subgraph(l1.node);
    bool ok3b = ag::inplace::recompute_inplace(l3.node);
    std::cout << "  After aggressive deletion:\n";
    std::cout << "    Layer1 recompute: " << (ok1b ? "✅ success" : "⚠️ failed (metadata removed)") << "\n";
    std::cout << "    Layer3 recompute: " << (ok3b ? "✅ success" : "⚠️ failed (metadata removed)") << "\n";

    // -------------------------------------------------------------
    // 7️⃣ ALIAS CONSISTENCY CHECK
    // -------------------------------------------------------------
    std::cout << "\n[Alias Consistency Check]\n";
    if (allclose(l1.val(), l2.val()))
        std::cout << "✅ Alias values consistent after recompute.\n";
    else
        std::cout << "❌ Alias values diverged.\n";

    // -------------------------------------------------------------
    // 8️⃣ FINAL VERSION TABLE
    // -------------------------------------------------------------
    std::cout << "\n[Final Version Table]\n";
    ag::inplace::debug::print_version_table();

    std::cout << "\n===== COMPLEX CAREFUL DELETION TEST COMPLETED =====\n";
    return 0;
}


// #include <iostream>
// #include <cmath>
// #include "ad/ag_all.hpp"
// #include "ad/autodiff/inplace.hpp"
// #include "ad/autodiff/careful_deletion.hpp"
// #include <iostream>
// #include <cmath>

// // Main framework headers
// #include "ad/ag_all.hpp"
// #include "ad/autodiff/inplace.hpp"
// #include "ad/autodiff/careful_deletion.hpp"

// using namespace ag;

// // Helper function to compare tensors using the correct data access API
// bool allclose(const Tensor& A, const Tensor& B, float tol = 0.5f) {
//     if (A.shape().dims != B.shape().dims) return false;
    
//     // Tensors must be on CPU to access data directly
//     Tensor A_cpu = A.to_cpu();
//     Tensor B_cpu = B.to_cpu();
    
//     const float* a_data = A_cpu.data<float>();
//     const float* b_data = B_cpu.data<float>();
    
//     for (size_t i = 0; i < A.numel(); ++i) {
//         if (std::fabs(a_data[i] - b_data[i]) > tol) {
//             std::cerr << "Mismatch at index " << i << ": " << a_data[i] << " vs " << b_data[i] << std::endl;
//             return false;
//         }
//     }
//     return true;
// }

// int main() {
//     std::cout << "===== COMPLEX CAREFUL DELETION TEST =====\n";

//     // -------------------------------------------------------------
//     // 1️⃣ BUILD A COMPLEX COMPUTATION GRAPH
//     // -------------------------------------------------------------
//     auto opts_param = TensorOptions().with_req_grad(true);
//     Tensor Ta = Tensor::randn(Shape{{3, 3}}, opts_param);
//     Tensor Tb = Tensor::randn(Shape{{3, 3}}, opts_param);
//     Tensor Tc = Tensor::randn(Shape{{3, 3}}, opts_param);
//     Tensor Td = Tensor::randn(Shape{{3, 3}}, opts_param);

//     Value a = make_tensor(Ta, "a");
//     Value b = make_tensor(Tb, "b");
//     Value c = make_tensor(Tc, "c");
//     Value d = make_tensor(Td, "d");

//     // Layer 1
//    // --- FIX: Checkpoint the parent node as well ---
    
//     // Layer 1
//     Value l1_parent = add(matmul(a, b), c);
//     l1_parent = checkpoint(l1_parent, CheckpointOptions{}); // MARK THE PARENT
    
//     Value l1 = relu(l1_parent);
//     l1 = checkpoint(l1, CheckpointOptions{}); // MARK THE CHILD
    
//     // --- END FIX ---

//     // Layer 2 - creates an alias intentionally
//     Value l2 = l1; // alias to l1
//     // Explicitly register alias to test the inplace subsystem
//     ag::inplace::register_tensor_alias((void*)l1.val().data(), l1.node.get());
//     ag::inplace::register_tensor_alias((void*)l2.val().data(), l2.node.get());

//     // Layer 3
//     Value l3 = add(matmul(l2, d), c);
//     l3 = inplace_checkpoint(l3); // Mark for in-place checkpointing
//     // Note: inplace_checkpoint() already calls mark_inplace_checkpoint internally.
    
//     // Layer 4 - nonlinearity
//     Value l4 = relu(add(l3, c));

//     // Output layer: sum of all elements
//     Value loss = sum(l4);

//     std::cout << "[Graph Build] Completed forward graph construction.\n";

//     // -------------------------------------------------------------
//     // 2️⃣ FORWARD AND BACKWARD
//     // -------------------------------------------------------------
//     backward(loss);
//     std::cout << "\n[Backward] Gradients computed successfully.\n";

//     debug::print_grad("a.grad", a);
//     debug::print_grad("b.grad", b);
//     debug::print_grad("c.grad", c);
//     debug::print_grad("d.grad", d);

//     // -------------------------------------------------------------
//     // 3️⃣ VERSION TRACKING AFTER FORWARD
//     // -------------------------------------------------------------
//     std::cout << "DEBUG: About to get version for l1.node@" << l1.node.get() << std::endl;
//     size_t v1 = inplace::get_tensor_version(l1.node.get());
//     std::cout << "DEBUG: Got version v1=" << v1 << std::endl;
    
//     std::cout << "DEBUG: About to get version for l3.node@" << l3.node.get() << std::endl;
//     size_t v3 = inplace::get_tensor_version(l3.node.get());
//     std::cout << "\n[Version Tracking]\n";
//     std::cout << "  Layer1 (checkpointed) version = " << v1 << "\n";
//     std::cout << "  Layer3 (inplace) version = " << v3 << "\n";

//     // Perform in-place modification to test version bump
//     std::cout << "\n[In-place Modification Test]\n";
//     l3.node->value += (Tensor::ones(l3.val().shape(), ag::options(l3.val())) * 0.1f);
//     inplace::bump_tensor_version(l3.node.get());
//     size_t v3_after = inplace::get_tensor_version(l3.node.get());
//     std::cout << "  Layer3 version after in-place update = " << v3_after << "\n";

//     // -------------------------------------------------------------
//     // 4️⃣ RUN CAREFUL DELETION (SAFE MODE)
//     // -------------------------------------------------------------
//     std::cout << "\n[Careful Deletion: SAFE MODE]\n";
//     ag::memory::sweep_safe_nodes(loss, ag::memory::DeletePolicy::AlwaysSafe);
//     std::cout << "  ✔ Safe deletion completed.\n";
//     std::cout << "  Check if checkpoint node is still protected: "
//               << (l1.node->is_checkpoint ? "✅ yes\n" : "❌ no\n");

//     // -------------------------------------------------------------
//     // 5️⃣ SIMULATE VALUE DROPS AND RECOMPUTE
//     // -------------------------------------------------------------
//     std::cout << "\n[Simulate Deallocation + Recomputation]\n";
//     // Correctly create empty tensors to simulate deallocation
//     l1.node->value = Tensor(Shape{}, TensorOptions{});
//     l3.node->value = Tensor(Shape{}, TensorOptions{});

//     bool ok1 = ag::checkpoint_impl::recompute_subgraph(l1.node);
//     // For in-place, the correct recompute function is different
//     bool ok3 = ag::inplace::recompute_inplace(l3.node); 
//     std::cout << "  Layer1 recompute: " << (ok1 ? "✅ success\n" : "❌ fail\n");
//     std::cout << "  Layer3 recompute: " << (ok3 ? "✅ success\n" : "❌ fail\n");

//     // TODO: Debug prints disabled due to tensor shape issues after recomputation
//     // The recomputation is succeeding but producing tensors with incorrect shapes
//     // This needs deeper investigation into the checkpoint/recompute logic
//     // if (ok1) {
//     //     debug::print_value("l1 (recomputed)", l1);
//     // }
//     // if (ok3) {
//     //     debug::print_value("l3 (recomputed)", l3);
//     // }

//     // -------------------------------------------------------------
//     // 6️⃣ CAREFUL DELETION (AGGRESSIVE MODE)
//     // -------------------------------------------------------------
//     std::cout << "\n[Careful Deletion: AGGRESSIVE MODE]\n";
//     ag::memory::sweep_safe_nodes(loss, ag::memory::DeletePolicy::Aggressive);

//     bool ok1b = ag::checkpoint_impl::recompute_subgraph(l1.node);
//     bool ok3b = ag::inplace::recompute_inplace(l3.node);
//     std::cout << "  After aggressive deletion:\n";
//     std::cout << "    Layer1 recompute: " << (ok1b ? "✅ success" : "⚠️ failed (metadata removed)") << "\n";
//     std::cout << "    Layer3 recompute: " << (ok3b ? "✅ success" : "⚠️ failed (metadata removed)") << "\n";

//     // -------------------------------------------------------------
//     // 7️⃣ ALIAS CONSISTENCY CHECK
//     // -------------------------------------------------------------
//     std::cout << "\n[Alias Consistency Check]\n";
//     // TODO: Disabled due to tensor shape issues after recomputation
//     // if (allclose(l1.val(), l2.val()))
//     //     std::cout << "✅ Alias values consistent after recompute.\n";
//     // else
//     //     std::cout << "❌ Alias values diverged.\n";
//     std::cout << "⚠️  Alias check skipped (tensor shape issues)\n";

//     // -------------------------------------------------------------
//     // 8️⃣ FINAL VERSION TABLE
//     // -------------------------------------------------------------
//     std::cout << "\n[Final Version Table]\n";
//     ag::inplace::debug::print_version_table();

//     std::cout << "\n===== COMPLEX CAREFUL DELETION TEST COMPLETED =====\n";
//     return 0;
// }