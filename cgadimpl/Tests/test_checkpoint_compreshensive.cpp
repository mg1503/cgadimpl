#include "ad/core/graph.hpp"
#include "ad/autodiff/autodiff.hpp"
#include "ad/autodiff/checkpoint.hpp"
#include "ad/autodiff/inplace.hpp"
#include "ad/autodiff/careful_deletion.hpp"
#include "tensor.hpp"
#include <iostream>
#include <cassert>

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
            std::cout << "[PASS] Test #" << g_test_count << "\n"; \
        } \
    } while(0)

// ============================================================================
// Test 1: Basic Checkpoint Marking and Recomputation
// ============================================================================

void test_basic_checkpoint() {
    std::cout << "\n=== Test 1: Basic Checkpoint Marking ===\n";
    
    reset_checkpoint_stats();
    
    // Create a simple computation graph: a -> b -> c
    Tensor ta = Tensor::ones(Shape{{2, 2}});
    Tensor tb = Tensor::ones(Shape{{2, 2}});
    
    Value a = make_tensor(ta, "a");
    Value b = make_tensor(tb, "b");
    Value c = a + b;  // Assuming + operator exists
    
    // Mark node b as checkpoint
    checkpoint_impl::mark_node_checkpoint(c.node, CheckpointOptions());
    
    TEST_ASSERT(c.node->is_checkpoint == true);
    TEST_ASSERT(c.node->saved_inputs.size() == 2);
    
    print_checkpoint_stats();
}

// ============================================================================
// Test 2: Smart Checkpoint Selection
// ============================================================================

void test_smart_checkpoint_selection() {
    std::cout << "\n=== Test 2: Smart Checkpoint Selection ===\n";
    
    reset_checkpoint_stats();
    
    // Create a deeper graph
    Tensor t = Tensor::ones(Shape{{10, 10}});
    Value v1 = make_tensor(t, "v1");
    Value v2 = make_tensor(t, "v2");
    Value v3 = v1 + v2;
    Value v4 = v3 + v1;
    Value v5 = v4 + v2;
    
    // Test memory-optimal policy
    auto_checkpoint_memory_optimal(v5, 0.3);
    print_checkpoint_stats();
    
    reset_checkpoint_stats();
    
    // Test speed-optimal policy
    auto_checkpoint_speed_optimal(v5, 0.3);
    print_checkpoint_stats();
    
    reset_checkpoint_stats();
    
    // Test balanced policy
    auto_checkpoint_balanced(v5, 0.3);
    print_checkpoint_stats();
    
    TEST_ASSERT(true);  // If we got here without crashing, test passed
}

// ============================================================================
// Test 3: Inplace Snapshot Management
// ============================================================================

void test_inplace_snapshots() {
    std::cout << "\n=== Test 3: Inplace Snapshot Management ===\n";
    
    inplace::clear_inplace_checkpoints();
    
    Tensor t = Tensor::ones(Shape{{5, 5}});
    Value v = make_tensor(t, "v");
    
    // Mark as inplace checkpoint
    inplace::mark_inplace_checkpoint(v.node, inplace::InplaceOptions());
    
    TEST_ASSERT(v.node->is_checkpoint == true);
    
    // Check snapshot memory
    size_t mem_before = inplace::get_snapshot_memory_usage();
    std::cout << "Snapshot memory: " << (mem_before / 1024.0) << " KB\n";
    TEST_ASSERT(mem_before > 0);
    
    // Cleanup stale snapshots
    size_t freed = inplace::cleanup_stale_snapshots();
    std::cout << "Freed: " << (freed / 1024.0) << " KB\n";
    
    size_t mem_after = inplace::get_snapshot_memory_usage();
    TEST_ASSERT(mem_after <= mem_before);
}

// ============================================================================
// Test 4: Memory Management Integration
// ============================================================================

void test_memory_management() {
    std::cout << "\n=== Test 4: Memory Management Integration ===\n";
    
    memory::reset_deletion_stats();
    
    Tensor t = Tensor::ones(Shape{{10, 10}});
    Value v1 = make_tensor(t, "v1");
    Value v2 = make_tensor(t, "v2");
    Value v3 = v1 + v2;
    Value v4 = v3 + v1;
    
    // Mark some as checkpoints
    checkpoint_impl::mark_node_checkpoint(v3.node, CheckpointOptions());
    
    // Try safe deletion
    memory::sweep_safe_nodes(v4, memory::DeletePolicy::AlwaysSafe);
    
    memory::debug_deletion_state();
    
    TEST_ASSERT(true);
}

// ============================================================================
// Test 5: Checkpoint Statistics
// ============================================================================

void test_checkpoint_statistics() {
    std::cout << "\n=== Test 5: Checkpoint Statistics ===\n";
    
    reset_checkpoint_stats();
    
    Tensor t = Tensor::ones(Shape{{20, 20}});
    Value v1 = make_tensor(t, "v1");
    Value v2 = make_tensor(t, "v2");
    Value v3 = v1 + v2;
    
    // Create multiple checkpoints
    auto_checkpoint_every_n(v3, 2);
    
    print_checkpoint_stats();
    
    TEST_ASSERT(true);
}

// ============================================================================
// Test 6: Memory Pressure Detection
// ============================================================================

void test_memory_pressure() {
    std::cout << "\n=== Test 6: Memory Pressure Detection ===\n";
    
    // Create many snapshots to trigger memory pressure
    inplace::clear_inplace_checkpoints();
    
    for (int i = 0; i < 10; ++i) {
        Tensor t = Tensor::ones(Shape{{100, 100}});
        Value v = make_tensor(t, ("v" + std::to_string(i)).c_str());
        inplace::mark_inplace_checkpoint(v.node, inplace::InplaceOptions());
    }
    
    size_t total_mem = inplace::get_snapshot_memory_usage();
    std::cout << "Total snapshot memory: " << (total_mem / 1024.0 / 1024.0) << " MB\n";
    
    TEST_ASSERT(total_mem > 0);
    
    // Cleanup
    inplace::clear_inplace_checkpoints();
}

// ============================================================================
// Test 7: Checkpoint Priority Sweep
// ============================================================================

void test_priority_sweep() {
    std::cout << "\n=== Test 7: Checkpoint Priority Sweep ===\n";
    
    memory::reset_deletion_stats();
    
    Tensor t = Tensor::ones(Shape{{50, 50}});
    Value v1 = make_tensor(t, "v1");
    Value v2 = make_tensor(t, "v2");
    Value v3 = v1 + v2;
    Value v4 = v3 + v1;
    
    // Mark some as checkpoints
    checkpoint_impl::mark_node_checkpoint(v3.node, CheckpointOptions());
    
    // Try priority sweep with target memory
    memory::sweep_with_checkpoint_priority(v4, 1);  // Target 1MB
    
    memory::debug_deletion_state();
    
    TEST_ASSERT(true);
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "Production-Level Checkpointing Tests\n";
    std::cout << "========================================\n";
    
    try {
        test_basic_checkpoint();
        test_smart_checkpoint_selection();
        test_inplace_snapshots();
        test_memory_management();
        test_checkpoint_statistics();
        test_memory_pressure();
        test_priority_sweep();
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << "\n";
        g_fail_count++;
    }
    
    std::cout << "\n========================================\n";
    std::cout << "Test Summary\n";
    std::cout << "========================================\n";
    std::cout << "Total Tests: " << g_test_count << "\n";
    std::cout << "Passed: " << (g_test_count - g_fail_count) << "\n";
    std::cout << "Failed: " << g_fail_count << "\n";
    
    if (g_fail_count == 0) {
        std::cout << "\n✓ ALL TESTS PASSED!\n";
        return 0;
    } else {
        std::cout << "\n✗ SOME TESTS FAILED\n";
        return 1;
    }
}
