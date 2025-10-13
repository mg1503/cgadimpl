#include <iostream>
#include <cmath>
#include "ad/ag_all.hpp"
#include "ad/inplace.hpp"

using namespace ag;

int main() {
    std::cout << "===== Versioning System Test =====\n";

    Tensor T = Tensor::ones(2,2) * 5;
    Value v = param(T, "v");
    std::cout << "[Init] Tensor v:\n" << v.val();

    inplace::mark_inplace_checkpoint(v.node);
    size_t ver0 = inplace::get_tensor_version(v.node.get());
    std::cout << "[Step 0] Initial version = " << ver0 << "\n";
    inplace::debug::print_version_table();

    // 1st in-place modification
    std::cout << "\n[Step 1] Performing in-place add_\n";
    v.node->value.add_(Tensor::ones_like(v.val()));
    inplace::bump_tensor_version(v.node.get());
    size_t ver1 = inplace::get_tensor_version(v.node.get());
    std::cout << "  -> version = " << ver1 << "\n";
    inplace::debug::print_version_table();

    // 2nd in-place modification (add_ again or mul_ if available)
    std::cout << "\n[Step 2] Performing another in-place add_\n";
    v.node->value.add_(Tensor::ones(2,2)*2);
    inplace::bump_tensor_version(v.node.get());
    size_t ver2 = inplace::get_tensor_version(v.node.get());
    std::cout << "  -> version = " << ver2 << "\n";
    inplace::debug::print_version_table();

    // Simulate recompute
    std::cout << "\n[Step 3] Simulating recompute via on_recomputed\n";
    ag::inplace::on_recomputed(v.node.get());
    size_t ver3 = inplace::get_tensor_version(v.node.get());
    std::cout << "  -> version = " << ver3 << "\n";
    inplace::debug::print_version_table();

    std::cout << "\n[Summary] v0=" << ver0 << " v1=" << ver1 << " v2=" << ver2 << " v3=" << ver3 << "\n";
    if (ver3 > ver2 && ver2 > ver1 && ver1 > ver0)
        std::cout << "✅ Version numbers increased monotonically.\n";
    else
        std::cout << "❌ Versioning sequence incorrect!\n";

    std::cout << "===== Versioning System Test Completed =====\n";
    return 0;
}
