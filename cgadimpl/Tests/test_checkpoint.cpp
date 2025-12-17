#include <iostream>
#include <vector>
#include <deque>
#include <unordered_set>
#include "ad/ag_all.hpp"
#include "ad/autodiff/checkpoint.hpp" // For checkpoint_impl::mark_node_checkpoint

using namespace ag;

int main() {
    std::cout << "===== mark_node_checkpoint Test =====\n";

    // 1. Create some tensors and build a small graph
    auto opts_param = TensorOptions().with_req_grad(true);
    auto x = make_tensor(Tensor::randn(Shape{{2, 2}}, opts_param), "x");
    auto W = make_tensor(Tensor::randn(Shape{{2, 2}}, opts_param), "W");
    auto b = make_tensor(Tensor::randn(Shape{{2, 2}}, opts_param), "b");

    // Graph: y = relu(x @ W + b)
    Value y1 = matmul(x, W);
    y1.node->debug_name = "y1_matmul";
    Value y2 = add(y1, b);
    y2.node->debug_name = "y2_add";
    Value y3 = relu(y2);
    y3.node->debug_name = "y3_relu";
    Value loss = sum(y3);
    loss.node->debug_name = "loss";

    // 2. Manually mark a node as a checkpoint
    std::cout << "\nMarking node 'y2_add' as a checkpoint...\n";
    checkpoint_impl::mark_node_checkpoint(y2.node);

    // 3. Traverse the graph and print only the marked nodes
    std::cout << "\n--- Verifying Marked Checkpoints ---\n";
    int marked_count = 0;
    std::deque<std::shared_ptr<Node>> q;
    std::unordered_set<Node*> visited;
    q.push_back(loss.node);

    while (!q.empty()) {
        auto n = q.front(); q.pop_front();
        if (!n || visited.count(n.get())) continue;
        visited.insert(n.get());

        if (n->is_checkpoint) {
            std::cout << "Found marked checkpoint node: " << n->debug_name << " ✅\n";
            marked_count++;
        }

        for (auto &p : n->inputs) {
            if (p) q.push_back(p);
        }
    }

    if (marked_count == 0) {
        std::cout << "❌ No nodes were marked as checkpoints.\n";
    }

    std::cout << "\n===== Test Completed =====\n";
    return 0;
}