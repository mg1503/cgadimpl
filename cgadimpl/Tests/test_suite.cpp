//============================================================
// file: tests/test_checkpoint_suite.cpp
// purpose: Production-grade test suite for checkpoint_impl::mark_node_checkpoint
//============================================================

#include <iostream>
#include <vector>
#include <deque>
#include <unordered_set>
#include <cassert>
#include "ad/ag_all.hpp"
#include "ad/checkpoint.hpp"

using namespace ag;

// Helper to create a dummy tensor
static Tensor dummy_tensor() {
    return Tensor::ones(Shape{{1, 1}}, TensorOptions().with_req_grad(true));;
}

// Helper to run a test case
static void run_test(const std::string& test_name, std::function<void()> test_func) {
    std::cout << "[RUNNING] " << test_name << "..." << std::endl;
    test_func();
    std::cout << "[PASSED] " << test_name << std::endl;
}

int main() {
    std::cout << "===== mark_node_checkpoint Comprehensive Test Suite =====\n" << std::endl;

    // Test 1: Mark a simple node with one input
    run_test("Test 1: Mark a simple unary node", []() {
        auto x = make_tensor(dummy_tensor(), "x");
        Value y = relu(x);
        checkpoint_impl::mark_node_checkpoint(y.node);
        assert(y.node->is_checkpoint);
        assert(y.node->saved_inputs.size() == 1);
        std::cout<< x.node->debug_name << std::endl;
        std::cout<< y.node->debug_name << std::endl;
        assert(y.node->saved_inputs[0].node == x.node);
    });

    // Test 2: Mark a node with multiple inputs
    run_test("Test 2: Mark a binary node (add)", []() {
        auto x = make_tensor(dummy_tensor(), "x");
        Value w = make_tensor(dummy_tensor(), "w");
        Value y = add(x, w);
        checkpoint_impl::mark_node_checkpoint(y.node);
        assert(y.node->is_checkpoint);
        assert(y.node->saved_inputs.size() == 2);
        std::cout<< x.node->debug_name << std::endl;
        std::cout<< y.node->debug_name << std::endl;
        assert(y.node->saved_inputs[0].node == x.node);
        assert(y.node->saved_inputs[1].node == w.node);
    });

    // Test 3: Attempt to mark a leaf node
    run_test("Test 3: Attempt to mark a leaf node", []() {
        auto x = make_tensor(dummy_tensor(), "x");
        checkpoint_impl::mark_node_checkpoint(x.node);
        assert(x.node->is_checkpoint);
        assert(x.node->saved_inputs.empty()); // Leaf has no inputs to save
    });

    // Test 4: Attempt to mark a null node
    run_test("Test 4: Mark a null shared_ptr", []() {
        std::shared_ptr<Node> null_node = nullptr;
        checkpoint_impl::mark_node_checkpoint(null_node);
        // Should not crash
    });

    // Test 5: Mark a node that is already a checkpoint (idempotency)
    run_test("Test 5: Mark an already-checkpointed node", []() {
        Value x = make_tensor(dummy_tensor(), "x");
        Value y = relu(x);
        checkpoint_impl::mark_node_checkpoint(y.node); // First time, state is correct.
        assert(y.node->saved_inputs.size() == 1);

        // The implementation is idempotent. If called again, it should just return.
        checkpoint_impl::mark_node_checkpoint(y.node); 
        assert(y.node->is_checkpoint);
        // The state should remain correct.
        assert(y.node->saved_inputs.size() == 1);
    });

    // Test 6: Mark a node in a chain (y = relu(relu(x)))
    run_test("Test 6: Mark a node in a chain", []() {
        Value x = make_tensor(dummy_tensor(), "x");
        Value y1 = relu(x);
        Value y2 = relu(y1);
        checkpoint_impl::mark_node_checkpoint(y2.node);
        assert(y2.node->is_checkpoint);
        assert(y2.node->saved_inputs.size() == 1);
        assert(y2.node->saved_inputs[0].node == y1.node);
    });

    // Test 7: Mark a node that is an input to multiple children
    run_test("Test 7: Mark a node with multiple children (branch)", []() {
        Value x = make_tensor(dummy_tensor(), "x");
        checkpoint_impl::mark_node_checkpoint(x.node); // Mark the shared parent
        Value y1 = relu(x);
        Value y2 = relu(x); // x is input to y1 and y2
        assert(x.node->is_checkpoint);
    });

    // Test 8: Mark a node in a diamond-shaped graph
    run_test("Test 8: Mark a node in a diamond graph", []() {
        Value x = make_tensor(dummy_tensor(), "x");
        Value y1 = relu(x);
        Value y2 = relu(x);
        Value z = add(y1, y2);
        checkpoint_impl::mark_node_checkpoint(y1.node);
        assert(y1.node->is_checkpoint);
        assert(y1.node->saved_inputs.size() == 1);
        assert(y1.node->saved_inputs[0].node == x.node);
    });

    // Test 9: Mark the final node in a diamond-shaped graph
    run_test("Test 9: Mark the final node of a diamond graph", []() {
        Value x = make_tensor(dummy_tensor(), "x");
        Value y1 = relu(x);
        Value y2 = relu(x);
        Value z = add(y1, y2);
        checkpoint_impl::mark_node_checkpoint(z.node);
        assert(z.node->is_checkpoint);
        assert(z.node->saved_inputs.size() == 2);
        assert(z.node->saved_inputs[0].node == y1.node);
        assert(z.node->saved_inputs[1].node == y2.node);
    });

    // Test 10: Mark a node with a null input pointer in its inputs list
    run_test("Test 10: Mark a node with a null input", []() {
        Value x = make_tensor(dummy_tensor(), "x");
        auto y_node = std::make_shared<Node>(dummy_tensor(), Op::Add, false, "y");
        y_node->inputs = {x.node, nullptr}; // Manually create a null input
        Value y(y_node);

        checkpoint_impl::mark_node_checkpoint(y.node);
        assert(y.node->is_checkpoint);
        assert(y.node->saved_inputs.size() == 2);
        assert(y.node->saved_inputs[0].node == x.node);
        assert(y.node->saved_inputs[1].node == nullptr); // Should correctly save the null
    });

    // Test 11: Verify reference count increase on parents
    run_test("Test 11: Verify parent reference count increases", []() {
        std::shared_ptr<Node> x_node;
        long initial_ref_count = 0;
        long final_ref_count = 0;

        {
            Value x = make_tensor(dummy_tensor(), "x");
            x_node = x.node;
            initial_ref_count = x_node.use_count(); // Refs: x, x_node

            Value y = relu(x);
            checkpoint_impl::mark_node_checkpoint(y.node);
            // After marking, y.node->saved_inputs holds another shared_ptr to x_node
            final_ref_count = x_node.use_count();
        }
        // y goes out of scope, releasing its hold on x_node via `inputs` and `saved_inputs`
        assert(final_ref_count > initial_ref_count);
    });

    // Test 12: Verify reference count logic on destruction
    run_test("Test 12: Verify reference count decreases on destruction",  [](){
        Value x = make_tensor(dummy_tensor(), "x");
        std::weak_ptr<Node> weak_x = x.node;

        {
            Value y = relu(x);
            checkpoint_impl::mark_node_checkpoint(y.node);
            assert(!weak_x.expired()); // x should be alive
        } // y is destroyed here, releasing its shared_ptrs to x.node

        // After y is destroyed, only the original `x` Value holds a ref.
        // If there was a leak, the count would be > 1.
        // Note: This is an indirect check. A true leak would keep it alive even after `x` is gone.
        assert(x.node.use_count() == 1);
    });

    // Test 13: Mark a node with many inputs
    run_test("Test 13: Mark a node with many inputs", []() {
        auto n = std::make_shared<Node>(dummy_tensor(), Op::Add, false, "multi_add");
        const int num_inputs = 100;
        std::vector<Value> parents;
        for (int i = 0; i < num_inputs; ++i) {
            Value p = make_tensor(dummy_tensor());
            parents.push_back(p);
            n->inputs.push_back(p.node);
        }
        checkpoint_impl::mark_node_checkpoint(n);
        assert(n->is_checkpoint);
        assert(n->saved_inputs.size() == num_inputs);
    });

    // Test 14: Mark a node with no inputs (other than leaf)
    run_test("Test 14: Mark a non-leaf node with no inputs", []() {
        auto n = std::make_shared<Node>(dummy_tensor(), Op::Add, false, "no_inputs");
        n->inputs.clear();
        checkpoint_impl::mark_node_checkpoint(n);
        assert(n->is_checkpoint);
        assert(n->saved_inputs.empty());
    });

    // Test 15: Ensure `saved_inputs` is cleared before saving
    run_test("Test 15: Ensure saved_inputs is cleared first", []() {
        Value x = make_tensor(dummy_tensor(), "x");
        Value y = relu(x);
        // Manually add a fake saved input
        y.node->saved_inputs.emplace_back();
        
        checkpoint_impl::mark_node_checkpoint(y.node);
        assert(y.node->is_checkpoint);
        assert(y.node->saved_inputs.size() == 1); // Should be 1, not 2
        assert(y.node->saved_inputs[0].node == x.node);
    });

    // Test 16: Mark a node from a complex operation (e.g., 4 inputs)
    run_test("Test 16: Mark a node from a 4-input op", []() {
        Value a = make_tensor(dummy_tensor());
        Value b = make_tensor(dummy_tensor());
        Value c = make_tensor(dummy_tensor());
        Value d = make_tensor(dummy_tensor());
        Value y = attention(a, b, c, d);
        checkpoint_impl::mark_node_checkpoint(y.node);
        assert(y.node->is_checkpoint);
        assert(y.node->saved_inputs.size() == 4);
    });

    // Test 17: Mark a node created with a float literal input
    run_test("Test 17: Mark a node with a float literal input", []() {
        Value x = make_tensor(dummy_tensor());
        Value y = leaky_relu(x, 0.1f); // Creates a constant node for 0.1f
        checkpoint_impl::mark_node_checkpoint(y.node);
        assert(y.node->is_checkpoint);
        assert(y.node->saved_inputs.size() == 2);
        assert(y.node->saved_inputs[0].node == x.node);
        assert(y.node->saved_inputs[1].node->op == Op::Leaf); // The literal
    });

    // Test 18: Mark a node deep in a graph
    run_test("Test 18: Mark a node deep in a graph", []() {
        Value v = make_tensor(dummy_tensor());
        for (int i = 0; i < 50; ++i) {
            v = relu(v);
        }
        checkpoint_impl::mark_node_checkpoint(v.node);
        assert(v.node->is_checkpoint);
        assert(v.node->saved_inputs.size() == 1);
    });

    // Test 19: Mark a node whose input is also a checkpoint
    run_test("Test 19: Mark a node whose parent is a checkpoint", []() {
        Value x = make_tensor(dummy_tensor());
        Value y1 = relu(x);
        Value y2 = relu(y1);
        checkpoint_impl::mark_node_checkpoint(y1.node); // Mark parent
        checkpoint_impl::mark_node_checkpoint(y2.node); // Mark child
        assert(y1.node->is_checkpoint);
        assert(y2.node->is_checkpoint);
        assert(y2.node->saved_inputs[0].node == y1.node);
    });

    // Test 20: Memory leak sanity check
    run_test("Test 20: Memory leak sanity check", []() {
        std::weak_ptr<Node> weak_x, weak_y;
        {
            Value x = make_tensor(dummy_tensor());
            Value y = relu(x);
            weak_x = x.node;
            weak_y = y.node;
            checkpoint_impl::mark_node_checkpoint(y.node);
        } // All values go out of scope here.

        // If shared_ptrs in saved_inputs created a cycle, these would not expire.
        assert(weak_x.expired());
        assert(weak_y.expired());
    });

    // Test 21: Demonstrate memory being held by a checkpoint
    run_test("Test 21: Demonstrate memory hold by checkpoint", []() {
        Value x = make_tensor(dummy_tensor(), "x");
        std::weak_ptr<Node> weak_x = x.node; // Create a weak reference to watch x
        Value z; // Declare z in the outer scope so it stays alive.
    
        {
            z = relu(x);
            checkpoint_impl::mark_node_checkpoint(z.node); // z now holds a SAVED reference to x.
    
            x = Value(); // Release the original handle to the 'x' node.
                         // z's saved_inputs should now be the only thing keeping x.node alive.
        } // The inner scope closes, but z is NOT destroyed.
    
        assert(!weak_x.expired() && "Node 'x' should NOT be deallocated because a checkpoint's saved_inputs still refers to it.");
    });

    // Test 22: Demonstrate recompute failure on a broken graph chain
    run_test("Test 22: Recompute failure on broken graph", []() {
        Value x = make_tensor(dummy_tensor(), "x");
        Value y1 = relu(x);
        y1.node->debug_name = "y1";
        Value y2 = relu(y1);
        y2.node->debug_name = "y2";

        // Checkpoint the final node. This saves a reference to y1.
        // The saved reference is what we will break.
        checkpoint_impl::mark_node_checkpoint(y2.node);

        // Manually break the graph structure by replacing the saved parent with a nullptr.
        // This simulates a catastrophic failure where the parent node is completely lost.
        std::cout << "  > Manually breaking graph link: y2 -> y1" << std::endl;
        y2.node->saved_inputs[0].node = nullptr;

        // Simulate that the intermediate value for y2 was freed.
        y2.node->value.reset();

        // Attempt to recompute. This should fail.
        bool recompute_ok = checkpoint_impl::recompute_subgraph(y2.node);
        assert(!recompute_ok && "Recomputation should fail because the graph structure was broken.");
    });

    std::cout << "\n===== All 22 tests passed successfully! =====\n" << std::endl;
    return 0;
}