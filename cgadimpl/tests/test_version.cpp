#include <iostream>
#include <cmath>
#include "ad/ag_all.hpp"
#include "ad/inplace.hpp"
#include "ad/debug.hpp" // --- FIX: Include for printing ---

using namespace ag;
using namespace OwnTensor; // For Shape, TensorOptions

int main() {
    std::cout << "===== Versioning System Test =====\n";

    // --- FIX: Use modern factories ---
    Tensor T = Tensor::ones(Shape{{2, 2}}) * 5.0f;
    Value v = make_tensor(T, "v");
    ag::debug::print_value("Initial Tensor v", v);

    inplace::mark_inplace_checkpoint(v.node);
    size_t ver0 = inplace::get_tensor_version(v.node.get());
    std::cout << "[Step 0] Initial version = " << ver0 << "\n";
    inplace::debug::print_version_table();

    // 1st in-place modification
    std::cout << "\n[Step 1] Performing in-place add\n";
    // --- FIX: Use += and modern factories ---
    v.node->value += Tensor::ones(v.val().shape(), ag::options(v.val()));
    inplace::bump_tensor_version(v.node.get());
    size_t ver1 = inplace::get_tensor_version(v.node.get());
    std::cout << "  -> version = " << ver1 << "\n";
    inplace::debug::print_version_table();

    // 2nd in-place modification
    std::cout << "\n[Step 2] Performing another in-place add\n";
    // --- FIX: Use += and modern factories ---
    v.node->value += (Tensor::ones(Shape{{2, 2}}) * 2.0f);
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