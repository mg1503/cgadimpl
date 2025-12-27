// =====================
// file: cgadimpl/src/core/graph.cpp
// =====================
#include "ad/core/graph.hpp"
#include <iostream>
#include <stack>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>

namespace ag {

// ==========================================
// Graph Traversal (Iterative DFS)
// ==========================================
std::vector<Node*> topo_from(Node* root) {
    if (!root) return {};

    std::cerr << "[DEBUG topo_from] Starting with root=" << root << std::endl;
    std::vector<Node*> order;
    std::unordered_set<Node*> visited;
    
    // We'll use a standard iterative approach for topological sort.
    // We want a topological sort that respects dependencies:
    // If A depends on B (A has input B), B must come before A in the result.
    // This allows forward execution.
    // However, for BACKWARD pass (which is usually what topo_order is used for in current context),
    // we want Root first, then children?
    // Actually, `autodiff.cpp` typically iterates the list. S
    // Standard DFS Post-Order gives [DeepestDependency, ..., Root].
    // If we want [Root, ..., DeepestDependency], we reverse it.
    // Let's implement standard Post-Order first.
    
    struct Frame {
        Node* node;
        size_t next_edge_idx = 0;
    };
    
    std::vector<Frame> call_stack;
    call_stack.push_back({root, 0});
    
    int iterations = 0;
    while (!call_stack.empty()) {
        iterations++;
        if (iterations % 1000 == 0) {
            std::cerr << "[DEBUG topo_from] iter=" << iterations << ", stack_size=" << call_stack.size() << ", visited=" << visited.size() << ", order=" << order.size() << std::endl;
        }
        if (iterations > 100000) {
            std::cerr << "[ERROR topo_from] Too many iterations! Possible cycle or bug. Breaking." << std::endl;
            break;
        }
        
        Frame& frame = call_stack.back();
        Node* u = frame.node;
        
        if (frame.next_edge_idx == 0) {
            if (visited.count(u)) {
                call_stack.pop_back();
                continue;
            }
            visited.insert(u);
        }
        
        // Process next child
        bool pushed_child = false;
        while (frame.next_edge_idx < u->next_edges.size()) {
            // CRITICAL: Don't use a reference here! call_stack.push_back() can reallocate the vector
            // making any references invalid. Access by index or copy the pointer.
            Node* child = u->next_edges[frame.next_edge_idx].function.get();
            frame.next_edge_idx++;
            
            if (child && visited.find(child) == visited.end()) {
                call_stack.push_back({child, 0});
                pushed_child = true;
                break; // Process this child first
            }
        }
        
        if (!pushed_child) {
            // All children processed
            order.push_back(u);
            call_stack.pop_back();
        }
    }
    
    std::cerr << "[DEBUG topo_from] Completed. total_nodes=" << order.size() << std::endl;
    // Note: This returns Post-Order (Inputs -> Outputs).
    // If the caller expects Output -> Inputs (for backward), they usually iterate in reverse
    // OR this function should reverse it.
    // Looking at `autodiff.cpp`:
    // `auto order = topo_from(root.node.get());`
    // `for (auto it = order.rbegin(); it != order.rend(); ++it)` -> Reverse of this order.
    // So if this is Post-Order (Inputs...Root), then rbegin (Root...Inputs) is correct for Backward.
    
    return order;
}

} // namespace ag