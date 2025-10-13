// #include <iostream>
// #include "ad/ag_all.hpp"
// #include <memory>
// #include "ad/careful_deletion.hpp"

// using namespace ag;

// int main() {
//     std::cout << "===== Careful Deletion System Test =====\n";

//     // --------------------------------------------------------------
//     // 1️⃣ Build a simple computation graph
//     // y = sum( relu(a @ b + c) )
//     // --------------------------------------------------------------
//     Tensor Ta = Tensor::randn(3, 3, 42);
//     Tensor Tb = Tensor::randn(3, 3, 7);
//     Tensor Tc = Tensor::randn(3, 3, 99);

//     Value a = param(Ta, "a");
//     Value b = param(Tb, "b");
//     Value c = param(Tc, "c");

//     Value z = add(matmul(a, b), c);      // intermediate
//     z = checkpoint(z, CheckpointOptions{});                   // mark z as gradient checkpoint
//     z = inplace_checkpoint(z);           // also protect inplace recomputation

//     Value y = sum(relu(z));              // output scalar

//     // --------------------------------------------------------------
//     // 2️⃣ Run forward and backward
//     // --------------------------------------------------------------
//     backward(y);

//     std::cout << "\n[Before Careful Deletion]\n";
//     std::cout << "a.grad():\n" << a.grad();
//     std::cout << "b.grad():\n" << b.grad();
//     std::cout << "c.grad():\n" << c.grad();

//     // --------------------------------------------------------------
//     // 3️⃣ Run a memory sweep (safe deletion)
//     // --------------------------------------------------------------
//     std::cout << "\n[Running Careful Deletion Sweep]\n";
//     ag::memory::sweep_safe_nodes(y, ag::memory::DeletePolicy::AlwaysSafe);

//     // --------------------------------------------------------------
//     // 4️⃣ Try recomputing checkpointed node (should still work)
//     // --------------------------------------------------------------
//     std::cout << "\n[Recomputing Checkpointed Node]\n";
//     bool ok = ag::checkpoint_impl::recompute_subgraph(z.node);
//     if (ok) {
//         std::cout << "✅ Recomputation succeeded after careful deletion.\n";
//         std::cout << "z.value (recomputed):\n" << z.val();
//     } else {
//         std::cout << "❌ Recomputation failed (checkpoint metadata missing).\n";
//     }

//     // --------------------------------------------------------------
//     // 5️⃣ Try deleting aggressively (for verification)
//     // --------------------------------------------------------------
//     std::cout << "\n[Running Aggressive Deletion]\n";
//     ag::memory::sweep_safe_nodes(y, ag::memory::DeletePolicy::Aggressive);

//     std::cout << "\n[After Aggressive Deletion]\n";
//     bool recompute_ok = ag::checkpoint_impl::recompute_subgraph(z.node);
//     if (recompute_ok)
//         std::cout << "✅ Recompute still succeeded (metadata preserved)\n";
//     else
//         std::cout << "⚠️  Recompute failed (snapshot removed aggressively)\n";

//     std::cout << "===== Careful Deletion Test Completed =====\n";
//     return 0;
// }

// #include <iostream>
// #include "ad/ag_all.hpp"
// #include "ad/careful_deletion.hpp"
// #include <ostream>
// #include "ad/inplace.hpp"

// using namespace ag;
// // Simple tolerance-based tensor comparison
// bool allclose(const Tensor& A, const Tensor& B, float tol = 1e-5f) {
//     if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
//     for (int i = 0; i < A.rows(); ++i)
//         for (int j = 0; j < A.cols(); ++j)
//             if (std::fabs(A(i,j) - B(i,j)) > tol)
//                 return false;
//     return true;
// }

// int main() {
//     std::cout << "===== Full Compatibility Test: Checkpoint + Inplace + Versioning + Alias + Careful Deletion =====\n";

//     // ----------------------------------------------------------------
//     // 1️⃣ Build a small graph with alias + checkpoint + inplace ops
//     // ----------------------------------------------------------------
//     Tensor Ta = Tensor::randn(2, 2, 42);
//     Tensor Tb = Tensor::randn(2, 2, 7);
//     Tensor Tc = Tensor::ones(2, 2) * 0.5;

//     Value a = param(Ta, "a");
//     Value b = param(Tb, "b");
//     Value c = param(Tc, "c");

//     Value z = add(matmul(a, b), c);   // intermediate node

//     // Alias to z (shares same data pointer)
//     Value z_alias = z;

//     // Register alias manually (for alias tracking)
//     inplace::register_tensor_alias((void*)z.val().data(), z.node.get());
//     inplace::register_tensor_alias((void*)z_alias.val().data(), z_alias.node.get());

//     // Mark as checkpoint and inplace checkpoint
//     z = checkpoint(z, CheckpointOptions{});
//     z = inplace_checkpoint(z);

//     std::cout << "[Init] Created computation graph with checkpointed node: " << z.node.get() << "\n";

//     // ----------------------------------------------------------------
//     // 2️⃣ Versioning system sanity check
//     // ----------------------------------------------------------------
//     size_t v0 = inplace::get_tensor_version(z.node.get());
//     std::cout << "[Version] Initial version: " << v0 << "\n";

//     // In-place modification to trigger version bump
//     z.node->value.add_(Tensor::ones_like(z.val()));
//     inplace::bump_tensor_version(z.node.get());
//     size_t v1 = inplace::get_tensor_version(z.node.get());
//     std::cout << "[Version] After in-place edit: " << v1 << "\n";

//     // ----------------------------------------------------------------
//     // 3️⃣ Forward + backward pass
//     // ----------------------------------------------------------------
//     Value y = sum(mul(relu(z), relu(z)));  // output loss

//     backward(y);

//     std::cout << "\n[Gradients after backward]\n";
//     std::cout << "a.grad:\n" << a.grad();
//     std::cout << "b.grad:\n" << b.grad();
//     std::cout << "c.grad:\n" << c.grad();

//     // ----------------------------------------------------------------
//     // 4️⃣ Check alias consistency
//     // ----------------------------------------------------------------
//     std::cout << "\n[Alias check]\n";
//     std::cout << "z.val:\n" << z.val();
//     std::cout << "z_alias.val:\n" << z_alias.val();
//     if (allclose(z.val(), z_alias.val()))
//         std::cout << "✅ Alias values consistent.\n";
//     else
//         std::cout << "❌ Alias values diverged.\n";

//     // ----------------------------------------------------------------
//     // 5️⃣ Trigger careful deletion (safe mode)
//     // ----------------------------------------------------------------
//     std::cout << "\n[Running Careful Deletion Sweep (Safe Mode)]\n";
//     ag::memory::sweep_safe_nodes(y, ag::memory::DeletePolicy::AlwaysSafe);

//     // Verify that checkpointed node survived safe deletion
//     if (z.node->is_checkpoint)
//         std::cout << "✅ Checkpoint node still protected after safe deletion.\n";
//     else
//         std::cout << "❌ Checkpoint flag lost during safe deletion.\n";

//     // ----------------------------------------------------------------
//     // 6️⃣ Manually drop z value, then recompute (checkpoint test)
//     // ----------------------------------------------------------------
//     std::cout << "\n[Recomputing Checkpointed Node]\n";
//     z.node->value = Tensor();  // simulate freeing memory

//     bool ok = ag::checkpoint_impl::recompute_subgraph(z.node);
//     if (ok) {
//         std::cout << "✅ Recomputation succeeded after deletion.\n";
//         std::cout << "z.value (recomputed):\n" << z.val();
//     } else {
//         std::cout << "❌ Recomputation failed (metadata missing).\n";
//     }

//     // ----------------------------------------------------------------
//     // 7️⃣ Aggressive deletion test
//     // ----------------------------------------------------------------
//     std::cout << "\n[Running Careful Deletion Sweep (Aggressive Mode)]\n";
//     ag::memory::sweep_safe_nodes(y, ag::memory::DeletePolicy::Aggressive);

//     std::cout << "\n[After Aggressive Deletion]\n";
//     bool recompute_ok = ag::checkpoint_impl::recompute_subgraph(z.node);
//     if (recompute_ok)
//         std::cout << "✅ Recompute still succeeded (metadata preserved)\n";
//     else
//         std::cout << "⚠️  Recompute failed (snapshot removed aggressively)\n";

//     // ----------------------------------------------------------------
//     // 8️⃣ Version after recompute
//     // ----------------------------------------------------------------
//     size_t v2 = inplace::get_tensor_version(z.node.get());
//     std::cout << "\n[Version] After recompute: " << v2 << "\n";
//     if (v2 > v1)
//         std::cout << "✅ Version incremented correctly after recompute.\n";
//     else
//         std::cout << "⚠️ Version not incremented — recompute may have reused snapshot.\n";

//     std::cout << "\n===== Full Compatibility Test Completed =====\n";
//     return 0;
// }

#include <iostream>
#include <cmath>
#include "ad/ag_all.hpp"
#include "ad/inplace.hpp"
#include "ad/careful_deletion.hpp"

using namespace ag;

// Simple tensor comparison for sanity checks
bool allclose(const Tensor& A, const Tensor& B, float tol = 1e-5f) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            if (std::fabs(A(i,j) - B(i,j)) > tol)
                return false;
    return true;
}

int main() {
    std::cout << "===== COMPLEX CAREFUL DELETION TEST =====\n";

    // -------------------------------------------------------------
    // 1️⃣ BUILD A COMPLEX COMPUTATION GRAPH
    // -------------------------------------------------------------
    Tensor Ta = Tensor::randn(3,3,1);
    Tensor Tb = Tensor::randn(3,3,2);
    Tensor Tc = Tensor::randn(3,3,3);
    Tensor Td = Tensor::randn(3,3,4);

    Value a = param(Ta, "a");
    Value b = param(Tb, "b");
    Value c = param(Tc, "c");
    Value d = param(Td, "d");

    // Layer 1
    Value l1 = relu(add(matmul(a,b), c));  // intermediate 1
    l1 = checkpoint(l1, CheckpointOptions{});                   // mark checkpoint

    // Layer 2 - creates an alias intentionally
    Value l2 = l1;                         // alias to l1
    ag::inplace::register_tensor_alias((void*)l1.val().data(), l1.node.get());
    ag::inplace::register_tensor_alias((void*)l2.val().data(), l2.node.get());

    // Layer 3
    Value l3 = add(matmul(l2, d), c);
    l3 = inplace_checkpoint(l3);           // in-place checkpoint
    ag::inplace::mark_inplace_checkpoint(l3.node);

    // Layer 4 - nonlinearity
    Value l4 = relu(add(l3, c));

    // Output layer: sum of squares
    Value loss = sum(mul(l4, l4));

    std::cout << "[Graph Build] Completed forward graph construction.\n";

    // -------------------------------------------------------------
    // 2️⃣ FORWARD AND BACKWARD
    // -------------------------------------------------------------
    backward(loss);
    std::cout << "\n[Backward] Gradients computed successfully.\n";

    std::cout << "a.grad:\n" << a.grad();
    std::cout << "b.grad:\n" << b.grad();
    std::cout << "c.grad:\n" << c.grad();
    std::cout << "d.grad:\n" << d.grad();

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
    l3.node->value.add_(Tensor::ones_like(l3.val()) * 0.1f);
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
    l1.node->value = Tensor();
    l3.node->value = Tensor();

    bool ok1 = ag::checkpoint_impl::recompute_subgraph(l1.node);
    bool ok3 = ag::checkpoint_impl::recompute_subgraph(l3.node);
    std::cout << "  Layer1 recompute: " << (ok1 ? "✅ success\n" : "❌ fail\n");
    std::cout << "  Layer3 recompute: " << (ok3 ? "✅ success\n" : "❌ fail\n");

    if (ok1) {
        std::cout << "  l1 (recomputed):\n" << l1.val();
    }
    if (ok3) {
        std::cout << "  l3 (recomputed):\n" << l3.val();
    }

    // -------------------------------------------------------------
    // 6️⃣ CAREFUL DELETION (AGGRESSIVE MODE)
    // -------------------------------------------------------------
    std::cout << "\n[Careful Deletion: AGGRESSIVE MODE]\n";
    ag::memory::sweep_safe_nodes(loss, ag::memory::DeletePolicy::Aggressive);

    bool ok1b = ag::checkpoint_impl::recompute_subgraph(l1.node);
    bool ok3b = ag::checkpoint_impl::recompute_subgraph(l3.node);
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

    std::cout << "===== COMPLEX CAREFUL DELETION TEST COMPLETED =====\n";
    return 0;
}
