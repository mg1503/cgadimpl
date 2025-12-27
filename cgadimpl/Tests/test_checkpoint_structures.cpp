#include "ad/graph.hpp"
#include "ad/autodiff.hpp"
#include "tensor.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <memory>
#include <limits>

using namespace ag;

int g_test_count = 0;
int g_fail_count = 0;

#define TEST_ASSERT(cond) \
    do { \
        g_test_count++; \
        if (!(cond)) { \
            std::cerr << "[FAIL] Test #" << g_test_count << " failed at line " << __LINE__ << ": " << #cond << "\n"; \
            g_fail_count++; \
        } else { \
             /* std::cout << "[PASS] Test #" << g_test_count << "\n"; */ \
        } \
    } while(0)

void test_initialization() {
    std::cout << "Running Initialization Tests...\n";
    Tensor t = Tensor::ones(Shape{{2, 2}});
    auto node = std::make_shared<Node>(t, Op::Leaf, true, "test_node");

    // 1. Check is_checkpoint default
    TEST_ASSERT(node->is_checkpoint == false);
    // 2. Check is_recomputed default
    TEST_ASSERT(node->is_recomputed == false);
    // 3. Check saved_memory_bytes default
    TEST_ASSERT(node->saved_memory_bytes == 0);
    // 4. Check recompute_id default
    TEST_ASSERT(node->recompute_id == 0);
    
    // 5. Check CheckpointPolicy enum values exist (compile-time check mainly, but runtime verify)
    TEST_ASSERT((int)CheckpointPolicy::None == 0);
    // 6. Check CheckpointPolicy::Memory
    TEST_ASSERT((int)CheckpointPolicy::Memory == 1);
    // 7. Check CheckpointPolicy::Speed
    TEST_ASSERT((int)CheckpointPolicy::Speed == 2);
    // 8. Check CheckpointPolicy::Auto
    TEST_ASSERT((int)CheckpointPolicy::Auto == 3);

    // 9. Verify node creation via make_tensor helper
    Value v = make_tensor(t, "helper_node");
    TEST_ASSERT(v.node->is_checkpoint == false);
    // 10. Verify helper node defaults
    TEST_ASSERT(v.node->recompute_id == 0);
}

void test_mutability() {
    std::cout << "Running Mutability Tests...\n";
    Tensor t = Tensor::ones({1});
    auto node = std::make_shared<Node>(t, Op::Leaf, false);

    // 11. Set is_checkpoint true
    node->is_checkpoint = true;
    TEST_ASSERT(node->is_checkpoint == true);
    // 12. Set is_checkpoint false
    node->is_checkpoint = false;
    TEST_ASSERT(node->is_checkpoint == false);

    // 13. Set is_recomputed true
    node->is_recomputed = true;
    TEST_ASSERT(node->is_recomputed == true);
    // 14. Toggle back
    node->is_recomputed = false;
    TEST_ASSERT(node->is_recomputed == false);

    // 15. Set saved_memory_bytes small
    node->saved_memory_bytes = 1024;
    TEST_ASSERT(node->saved_memory_bytes == 1024);
    // 16. Set saved_memory_bytes large
    node->saved_memory_bytes = 1024ULL * 1024 * 1024; // 1GB
    TEST_ASSERT(node->saved_memory_bytes == 1073741824ULL);

    // 17. Set recompute_id small
    node->recompute_id = 1;
    TEST_ASSERT(node->recompute_id == 1);
    // 18. Set recompute_id large
    node->recompute_id = 999999;
    TEST_ASSERT(node->recompute_id == 999999);
    
    // 19. Multiple field updates
    node->is_checkpoint = true;
    node->recompute_id = 50;
    TEST_ASSERT(node->is_checkpoint == true && node->recompute_id == 50);
    
    // 20. Reset all
    node->is_checkpoint = false;
    node->recompute_id = 0;
    TEST_ASSERT(node->is_checkpoint == false && node->recompute_id == 0);
}

void test_structural_integrity() {
    std::cout << "Running Structural Integrity Tests...\n";
    // 21. Check Node size is reasonable (not exploding). 
    // Previous Node size was roughly: Tensor(2) + vector(4) + ptrs + bools + padding.
    // We added 2 bools + size_t + uint64_t. roughly +24 bytes.
    // Just ensure it's > 0 and not massive.
    size_t node_size = sizeof(Node);
    TEST_ASSERT(node_size > 0);
    TEST_ASSERT(node_size < 1024); // Should be well under 1KB

    // 22. Alignment check (standard types usually aligned)
    TEST_ASSERT(alignof(Node) >= alignof(uint64_t));

    // 23. Check offset of new fields (ensure they are accessible)
    Node n;
    n.recompute_id = 0xDEADBEEF;
    uint64_t* ptr = &n.recompute_id;
    TEST_ASSERT(*ptr == 0xDEADBEEF);

    // 24. Check boolean packing (compiler dependent but usually efficient)
    // We can't strictly assert packing but we can assert distinct addresses or values
    n.is_checkpoint = true;
    n.is_recomputed = false;
    TEST_ASSERT(n.is_checkpoint != n.is_recomputed);

    // 25. Check no overlap
    n.saved_memory_bytes = 123;
    n.recompute_id = 456;
    TEST_ASSERT(n.saved_memory_bytes != n.recompute_id);
}

void test_interaction_existing() {
    std::cout << "Running Interaction Tests...\n";
    Tensor t = Tensor::ones({2, 2});
    auto node = std::make_shared<Node>(t, Op::Leaf, true);

    // 26. Check value shape preservation
    TEST_ASSERT(node->shape().size() == 2);
    // 27. Check requires_grad preservation
    TEST_ASSERT(node->requires_grad() == true);
    
    // 28. Modify checkpoint flag, check grad flag
    node->is_checkpoint = true;
    TEST_ASSERT(node->requires_grad() == true);

    // 29. Check op type
    TEST_ASSERT(node->op == Op::Leaf);
    
    // 30. Modify recompute, check op
    node->is_recomputed = true;
    TEST_ASSERT(node->op == Op::Leaf);

    // 31. Check inputs vector
    TEST_ASSERT(node->inputs.empty());
    
    // 32. Add input, check flags don't interfere
    auto input_node = std::make_shared<Node>(t, Op::Leaf, false);
    node->inputs.push_back(input_node);
    TEST_ASSERT(node->inputs.size() == 1);
    TEST_ASSERT(node->is_checkpoint == true);

    // 33. Check debug name
    node->debug_name = "my_node";
    TEST_ASSERT(std::string(node->debug_name) == "my_node");

    // 34. Check tape
    TEST_ASSERT(node->tape.empty());

    // 35. Check saved_rng
    TEST_ASSERT(node->has_saved_rng == false);
}

void test_graph_connectivity() {
    std::cout << "Running Graph Connectivity Tests...\n";
    Tensor t = Tensor::ones({1, 1});
    auto a = std::make_shared<Node>(t, Op::Leaf, true, "A");
    auto b = std::make_shared<Node>(t, Op::Leaf, true, "B");
    
    // 36. Link nodes
    b->inputs.push_back(a);
    
    // 37. Mark A as checkpoint
    a->is_checkpoint = true;
    TEST_ASSERT(b->inputs[0]->is_checkpoint == true);

    // 38. Mark B as recomputed
    b->is_recomputed = true;
    TEST_ASSERT(b->is_recomputed == true);

    // 39. Check A is not recomputed
    TEST_ASSERT(a->is_recomputed == false);

    // 40. Check traversal
    TEST_ASSERT(b->inputs[0].get() == a.get());
}

void test_smart_pointers() {
    std::cout << "Running Smart Pointer Tests...\n";
    Tensor t = Tensor::ones({1});
    
    // 41. Shared ptr ref count
    auto node = std::make_shared<Node>(t, Op::Leaf, true);
    TEST_ASSERT(node.use_count() == 1);

    // 42. Copy shared ptr
    auto node2 = node;
    TEST_ASSERT(node.use_count() == 2);
    
    // 43. Modify via copy
    node2->is_checkpoint = true;
    TEST_ASSERT(node->is_checkpoint == true);

    // 44. shared_from_this
    auto sft = node->shared_from_this();
    TEST_ASSERT(sft.get() == node.get());

    // 45. Weak ptr
    std::weak_ptr<Node> weak = node;
    TEST_ASSERT(!weak.expired());
}

void test_edge_cases() {
    std::cout << "Running Edge Case Tests...\n";
    Tensor t = Tensor::ones({1});
    auto node = std::make_shared<Node>(t, Op::Leaf, true);

    // 46. Max uint64_t recompute_id
    node->recompute_id = std::numeric_limits<uint64_t>::max();
    TEST_ASSERT(node->recompute_id == std::numeric_limits<uint64_t>::max());

    // 47. Max size_t saved_memory
    node->saved_memory_bytes = std::numeric_limits<size_t>::max();
    TEST_ASSERT(node->saved_memory_bytes == std::numeric_limits<size_t>::max());

    // 48. Zero size tensor with checkpoint flags
    Tensor empty = Tensor::ones({0});
    auto empty_node = std::make_shared<Node>(empty, Op::Leaf, false);
    empty_node->is_checkpoint = true;
    TEST_ASSERT(empty_node->is_checkpoint == true);

    // 49. Null debug name with flags
    auto unnamed = std::make_shared<Node>(t, Op::Leaf, false, nullptr);
    unnamed->is_recomputed = true;
    TEST_ASSERT(unnamed->is_recomputed == true);

    // 50. Self reference (circular) - just checking struct handles it (std::vector allows it)
    // Note: Actual circular graphs might leak memory if not handled, but struct definition allows it.
    // We just check we can add it.
    node->inputs.push_back(node);
    TEST_ASSERT(node->inputs[0] == node);
    node->inputs.clear(); // Break cycle for cleanup
}

int main() {
    std::cout << "Starting Phase 1 Verification: Core Data Structures\n";
    
    test_initialization();
    test_mutability();
    test_structural_integrity();
    test_interaction_existing();
    test_graph_connectivity();
    test_smart_pointers();
    // test_edge_cases();

    std::cout << "--------------------------------------------------\n";
    std::cout << "Total Tests: " << g_test_count << "\n";
    std::cout << "Passed: " << (g_test_count - g_fail_count) << "\n";
    std::cout << "Failed: " << g_fail_count << "\n";

    if (g_fail_count == 0) {
        std::cout << "ALL TESTS PASSED.\n";
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED.\n";
        return 1;
    }
}
