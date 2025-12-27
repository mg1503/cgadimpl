// =====================
// file: cgadimpl/src/core/graph.cpp
// =====================
#include "ad/core/graph.hpp"
#include <stack>
#include <unordered_set>
#include <algorithm>

namespace ag {

// ==========================================
// Graph Traversal (Iterative DFS)
// ==========================================
std::vector<Node*> topo_from(Node* root) {
    if (!root) return {};

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
    
    while (!call_stack.empty()) {
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
            Edge& e = u->next_edges[frame.next_edge_idx];
            frame.next_edge_idx++;
            
            if (e.function && visited.find(e.function.get()) == visited.end()) {
                call_stack.push_back({e.function.get(), 0});
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