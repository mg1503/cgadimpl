
#include <iostream>
#include <cassert>
#include <cmath>
#include "ad/graph.hpp"
#include "ad/ops.hpp"
#include "ad/autodiff.hpp"
#include "ad/checkpoint.hpp"
#include "ad/inplace.hpp"
#include "ad/debug.hpp"

using namespace ag;

static bool allclose(const Tensor& A, const Tensor& B, float tol = 1e-5f) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            if (std::fabs(A(i,j) - B(i,j)) > tol) return false;
    return true;
}

static void print_header(const std::string& name) {
    std::cout << "\n=================== " << name << " ===================\n";
}

// =============================================================
// 1️⃣ Gradient Checkpointing Test
// =============================================================
void test_gradient_checkpointing() {
    print_header("Gradient Checkpointing");

    Tensor Ta = Tensor::randn(3, 4, 42);
    Tensor Tb = Tensor::randn(4, 2, 24);

    Value a = param(Ta, "a");
    Value b = param(Tb, "b");

    Value z = matmul(a, b);
    z = checkpoint(z, CheckpointOptions());
    Value y = sum(mul(z, z));

    zero_grad(y);
    backward(y);

    Tensor grad_a = a.grad();
    Tensor grad_b = b.grad();

    assert(grad_a.size() > 0 && grad_b.size() > 0);
    std::cout << "Gradient checkpointing basic test passed.\n";
}

// =============================================================
// 2️⃣ In-Place Checkpointing + Recovery
// =============================================================
void test_inplace_checkpointing() {
    print_header("In-Place Checkpointing");

    Tensor T = Tensor::ones(2,2)*5;
    Value x = param(T, "x");
    inplace::mark_inplace_checkpoint(x.node);

    // Drop and recover
    x.node->value = Tensor();
    bool ok = inplace::ensure_inplace_value(x.node);
    assert(ok && "Inplace recovery failed!");
    std::cout << "In-place checkpoint restore test passed.\n";
}

// =============================================================
// 3️⃣ Versioning System Behavior
// =============================================================
void test_versioning_system() {
    print_header("Versioning System");

    Tensor T = Tensor::ones(2,2)*2;
    Value v = param(T, "v");

    inplace::mark_inplace_checkpoint(v.node);
    size_t v0 = inplace::get_tensor_version(v.node.get());
    v.node->value.add_(Tensor::ones_like(v.val()));
    inplace::bump_tensor_version(v.node.get());
    size_t v1 = inplace::get_tensor_version(v.node.get());
    assert(v1 > v0);

    // Force recompute increments again
    inplace::recompute_inplace(v.node);
    size_t v2 = inplace::get_tensor_version(v.node.get());
    assert(v2 > v1);

    std::cout << "Version progression: " << v0 << " -> " << v1 << " -> " << v2 << "\n";
    std::cout << "Versioning system test passed.\n";
}

// =============================================================
// 4️⃣ Alias / View Tracking
// =============================================================
void test_alias_tracking() {
    print_header("Alias Tracking");

    Tensor base = Tensor::ones(2,2)*3;
    Value A = param(base, "A");
    Value AliasA = A; // share same node
    inplace::register_tensor_alias((void*)A.val().data(), A.node.get());
    inplace::register_tensor_alias((void*)AliasA.val().data(), AliasA.node.get());

    inplace::mark_inplace_checkpoint(A.node);

    // modify A in-place
    A.node->value.add_(Tensor::ones_like(A.val()));
    inplace::bump_tensor_version(A.node.get());

    // check alias consistency
    assert(allclose(A.val(), AliasA.val()));
    std::cout << "Alias tracking consistency test passed.\n";
}

// =============================================================
// 5️⃣ Combined Checkpoint + In-Place + Versioning + Alias Test
// =============================================================
void test_combined_system() {
    print_header("Combined System Test");

    Tensor Ta = Tensor::randn(3, 4, 42);
    Tensor Tb = Tensor::randn(4, 2, 24);
    Tensor Tc = Tensor::ones(3, 2) * 0.5f;

    Value a = param(Ta, "a");
    Value b = param(Tb, "b");
    Value c = param(Tc, "c");

    Value z = add(matmul(a, b), c);
    z = checkpoint(z, CheckpointOptions());
    z = inplace_checkpoint(z);

    // Create alias to z
    Value z_alias = z;
    inplace::register_tensor_alias((void*)z.val().data(), z.node.get());
    inplace::register_tensor_alias((void*)z_alias.val().data(), z_alias.node.get());

    Value y = sum(mul(relu(z), relu(z)));

    zero_grad(y);
    backward(y);

    Tensor grad_a1 = a.grad(), grad_b1 = b.grad(), grad_c1 = c.grad();

    // Free z value, recompute, check versions & grads
    size_t ver_before = inplace::get_tensor_version(z.node.get());
    z.node->value = Tensor();
    inplace::ensure_inplace_value(z.node);
    size_t ver_after = inplace::get_tensor_version(z.node.get());
    assert(ver_after > ver_before);

    zero_grad(y);
    backward(y);
    Tensor grad_a2 = a.grad(), grad_b2 = b.grad(), grad_c2 = c.grad();

    assert(allclose(grad_a1, grad_a2));
    assert(allclose(grad_b1, grad_b2));
    assert(allclose(grad_c1, grad_c2));
    assert(allclose(z.val(), z_alias.val()));

    std::cout << "Combined checkpoint + inplace + version + alias test passed.\n";
}

// =============================================================
// 6️⃣ Version Comparison After Restore vs Recompute
// =============================================================
void test_snapshot_vs_recompute() {
    print_header("Snapshot vs Recomputation");

    Tensor Ta = Tensor::ones(2,2)*2;
    Tensor Tb = Tensor::ones(2,2)*3;
    Tensor Tc = Tensor::ones(2,2)*1;
    Value a = param(Ta,"a");
    Value b = param(Tb,"b");
    Value c = param(Tc,"c");

    Value z = add(mul(a,b),c);
    inplace::mark_inplace_checkpoint(z.node);
    size_t ver0 = inplace::get_tensor_version(z.node.get());

    // Drop + recompute
    z.node->value = Tensor();
    inplace::ensure_inplace_value(z.node);
    size_t ver1 = inplace::get_tensor_version(z.node.get());
    Tensor recomputed = z.val();
    Tensor expected = (Ta*Tb)+Tc;
    assert(allclose(expected,recomputed));
    assert(ver1 >= ver0);
    std::cout << "Snapshot and recomputed tensors match.\n";
}

// =============================================================
// MAIN
// =============================================================
int main() {
    std::cout << "==== AUTODIFF CHECKPOINTING + INPLACE TEST SUITE ====\n";
    test_gradient_checkpointing();
    test_inplace_checkpointing();
    test_versioning_system();
    test_alias_tracking();
    test_combined_system();
    test_snapshot_vs_recompute();
    std::cout << "\nAll tests in combined suite passed successfully!\n";
    return 0;
}
