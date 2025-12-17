// =========================================
// cgadimpl/src/tools/debug.cpp
// =========================================
#include "ad/utils/debug.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace ag::debug {

namespace {
bool  g_trace = false;
int   g_max_r = 6;
int   g_max_c = 8;
int   g_w     = 10;
int   g_prec  = 4;
// for backprop tracing
bool g_trace_bp = false;
bool g_trace_jvp = false;


std::string shape_str(const Tensor& t) {
    std::ostringstream os;
    const auto& dims = t.shape().dims;
    if (dims.empty()) {
        return "scalar";
    }
    for (size_t i = 0; i < dims.size(); ++i) {
        os << dims[i] << (i == dims.size() - 1 ? "" : "x");
    }
    return os.str();
}

} // anon

void enable_tracing(bool on) { g_trace = on; }


void print_tensor(const std::string& label, const Tensor& T){
    std::cout << label << " [" << shape_str(T) << "]\n";
    
    // The new tensor library has a powerful display method.
    // It automatically handles device placement (copies to CPU for printing) and formatting.
    // The second argument is the desired precision.
    T.display(std::cout, g_prec);
    
    std::cout << std::endl;
}
void print_value (const std::string& label, const Value& v){
    print_tensor(label, v.val());
}
void print_grad  (const std::string& label, const Value& v){
    print_tensor(label + ".grad", v.grad());
}

void on_node_created(const std::shared_ptr<Node>& n){
    if (!g_trace) return;
    std::ostringstream label;
    label << "[" << op_name(n->op) << "]"
          << (n->requires_grad() ? " (grad)" : "      ") // Use function call
          << "  value " << shape_str(n->value)          // Use new shape_str
          << "  @" << n.get();
    if (n->debug_name && n->debug_name[0] != '\0')
        label << "  name=\"" << n->debug_name << "\"";
    
    // We can't print the full tensor here as it might be huge.
    // Let's just print the label. The user can call print_value() for details.
    std::cout << label.str() << std::endl;
}

// ---- whole-graph printers ----

void print_all_values(const Value& root){
    auto order = topo_from(root.node.get());
    std::cout << "=== VALUES (topo) ===\n";
    for (Node* n : order) {
        std::ostringstream label;
        label << "[" << op_name(n->op) << "]"
              << (n->requires_grad() ? " (grad)" : "      ")
              << " value " << shape_str(n->value)
              << " @" << n;
        print_tensor(label.str(), n->value); // This will now use the powerful display()
    }
}

void print_all_grads(const Value& root){
    auto order = topo_from(root.node.get());
    std::cout << "=== GRADS (topo) ===\n";
    for (Node* n : order) if (n->requires_grad()) {
        std::ostringstream label;
        label << "[" << op_name(n->op) << "] grad " << shape_str(n->grad)
              << " @" << n;
        print_tensor(label.str(), n->grad); // This will now use the powerful display()
    }
}

void dump_dot(const Value& root, const std::string& filepath){
    auto order = topo_from(root.node.get());

    std::ofstream out(filepath);
    if (!out) {
        std::cerr << "debug::dump_dot: failed to open " << filepath << "\n";
        return;
    }

    out << "digraph AG {\n"
           "  rankdir=LR;\n"
           "  node [shape=record, fontsize=10];\n";

    // nodes
     for (Node* n : order) {
        std::ostringstream id; id << "n" << n;
        std::ostringstream lab;
        lab  << op_name(n->op) << "\\n" << shape_str(n->value); // Correct
        std::string color = (n->op==Op::Leaf ? (n->requires_grad() ? "lightgoldenrod1" : "lightgrey")
                                             : (n->requires_grad() ? "lightblue" : "white")); // Correct
        out << "  " << id.str()
            << " [label=\"" << lab.str() << "\", style=filled, fillcolor=\""
            << color << "\"];\n";
    }
    // edges
    for (Node* n : order) {
        std::ostringstream id; id << "n" << n;
        for (auto& pin : n->inputs) {
            out << "  n" << pin.get() << " -> " << id.str() << ";\n";
        }
    }

    out << "}\n";
    out.close();
    std::cout << "Wrote graph DOT to: " << filepath << "\n"
                 "Render with: dot -Tpng " << filepath << " -o build/graph.png\n";    

}
// ======================================================================
// Backprop/VJP graph (red arrows child->parent)
// ======================================================================


// --- public control ---
void enable_grad_tracing(bool on) { g_trace_bp = on; }

// --- backprop step hook ---
void on_backprop_step(Node* n, const Tensor& gy) {
    if (!g_trace_bp) return;
    auto shp = n->value.shape();
    std::cout << "[VJP] node @" << n << " op=" << op_name(n->op)
          << "  y_grad shape=" << shape_str(gy) << "\n";

    for (size_t k = 0; k < n->inputs.size(); ++k) {
        Node* p = n->inputs[k].get();
        // FIX: Use the new shape_str helper for N-D printing
        std::cout << "   -> parent[" << k << "] @" << p
                << " (" << op_name(p->op) << ") receives grad shape "
                << shape_str(p->value) << "\n";
    }
}

// --- VJP graph dump ---
void dump_vjp_dot(const Value& root, const std::string& filepath) {
    auto order = topo_from(root.node.get());
    std::ofstream out(filepath);
    if (!out) { std::cerr << "debug::dump_vjp_dot: failed to open " << filepath << "\n"; return; }

   
    out << "digraph VJP {\n"
           "  rankdir=LR;\n"
           "  node [shape=record, fontsize=10];\n";

    // forward nodes (same as dump_dot, but neutral colors)
    for (Node* n : order) {
        std::string color = (n->op==Op::Leaf ? (n->requires_grad() ? "lightgoldenrod1" : "lightgrey")
                                             : (n->requires_grad() ? "white" : "white"));
        out << "  n" << n
            << " [label=\"" << op_name(n->op) << "\\n" << shape_str(n->value)
            << "\", style=filled, fillcolor=\"" << color << "\"];\n";
    }
    // red VJP edges (child -> parent)
    for (Node* n : order) {
        for (auto& pin : n->inputs) {
            out << "  n" << n << " -> " << "n" << pin.get()
                << " [color=red, penwidth=1.5, label=\"grad\"];\n";
        }
    }

    out << "}\n";
    out.close();
    std::cout << "Wrote VJP DOT to: " << filepath << "\n"
                 "Render with: dot -Tpng " << filepath << " -o build/graph_vjp.png\n";
}
// ============================================================================
// JVP graph (green arrows parent->child)
// ============================================================================

void enable_jvp_tracing(bool on) { g_trace_jvp = on; }

void on_jvp_step(Node* n) {
    if (!g_trace_jvp) return;
    // FIX: Use the global shape_str helper
    std::cout << "[JVP] node @" << n
              << " op=" << op_name(n->op)
              << "  value=" << shape_str(n->value) << "\n";
    for (size_t k = 0; k < n->inputs.size(); ++k) {
        Node* p = n->inputs[k].get();
        // FIX: Use the global shape_str helper
        std::cout << "    parent[" << k << "] @" << p
                  << " (" << op_name(p->op) << ")  value="
                  << shape_str(p->value) << "\n";
    }
}

void dump_jvp_dot(const Value& root, const std::string& filepath) {
    auto order = topo_from(root.node.get());
    std::ofstream out(filepath);
    if (!out) { std::cerr << "debug::dump_jvp_dot: failed to open " << filepath << "\n"; return; }



    out << "digraph JVP {\n"
           "  rankdir=LR;\n"
           "  node [shape=record, fontsize=10];\n";
    for (Node* n : order) {
        out << "  n" << n << " [label=\""
            << op_name(n->op) << "\\n" << shape_str(n->value)
            << "\", style=filled, fillcolor=\"white\"];\n";
    }
    // Tangents flow forward: parent -> child (green)
    for (Node* n : order) {
        for (auto& pin : n->inputs) {
            out << "  n" << pin.get() << " -> n" << n
                << " [color=green, penwidth=1.5, label=\"tangent\"];\n";
        }
    }
    out << "}\n";
    out.close();
    std::cout << "Wrote JVP DOT to: " << filepath
              << "\nRender: dot -Tpng " << filepath << " -o build/graph_jvp.png\n";
}

} // namespace ag::debug
