#include "ad/debug.hpp"
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
    auto [r,c] = t.shape();
    std::ostringstream os; os << r << "x" << c;
    return os.str();
}

void print_tensor_impl(const Tensor& T) {
    int R = std::min(T.rows(), g_max_r);
    int C = std::min(T.cols(), g_max_c);
    std::cout << std::fixed << std::setprecision(g_prec);
    for (int i=0;i<R;++i) {
        for (int j=0;j<C;++j)
            std::cout << std::setw(g_w) << T(i,j);
        if (C < T.cols()) std::cout << "  ...";
        std::cout << "\n";
    }
    if (R < T.rows()) std::cout << "...\n";
}
} // anon

void enable_tracing(bool on) { g_trace = on; }
void set_print_limits(int max_rows, int max_cols, int width, int precision) {
    g_max_r = max_rows; g_max_c = max_cols; g_w = width; g_prec = precision;
}

void print_tensor(const std::string& label, const Tensor& T){
    std::cout << label << " [" << shape_str(T) << "]\n";
    print_tensor_impl(T);
    std::cout << std::endl;
}
void print_value (const std::string& label, const Value& v){
    print_tensor(label, v.val());
}
void print_grad  (const std::string& label, const Value& v){
    print_tensor(label + ".grad", v.node->grad);
}

void on_node_created(const std::shared_ptr<Node>& n){
    if (!g_trace) return;
    std::ostringstream label;
    label << "[" << op_name(n->op) << "]"
          << (n->requires_grad ? " (grad)" : "      ")
          << "  value " << shape_str(n->value)
          << "  @" << n.get();
    if (n->debug_name && n->debug_name[0] != '\0')
        label << "  name=\"" << n->debug_name << "\"";
    print_tensor(label.str(), n->value);
}

// ---- whole-graph printers ----

void print_all_values(const Value& root){
    auto order = topo_from(root.node.get());
    std::cout << "=== VALUES (topo) ===\n";
    for (Node* n : order) {
        std::ostringstream label;
        label << "[" << op_name(n->op) << "]"
              << (n->requires_grad ? " (grad)" : "      ")
              << " value " << shape_str(n->value)
              << " @" << n;
        print_tensor(label.str(), n->value);
    }
}

void print_all_grads(const Value& root){
    auto order = topo_from(root.node.get());
    std::cout << "=== GRADS (topo) ===\n";
    for (Node* n : order) if (n->requires_grad) {
        std::ostringstream label;
        label << "[" << op_name(n->op) << "] grad " << shape_str(n->grad)
              << " @" << n;
        print_tensor(label.str(), n->grad);
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
        std::ostringstream id;   id   << "n" << n;
        std::ostringstream lab;  lab  << op_name(n->op) << "\\n" << shape_str(n->value);
        std::string color = (n->op==Op::Leaf ? (n->requires_grad ? "lightgoldenrod1" : "lightgrey")
                                             : (n->requires_grad ? "lightblue" : "white"));
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
              << "  y_grad shape=" << shp.first << "x" << shp.second << "\n";

    // Show where gradients will go (just shapes; actual values printed by print_all_grads later)
    for (size_t k = 0; k < n->inputs.size(); ++k) {
        Node* p = n->inputs[k].get();
        auto pshp = p->value.shape();
        std::cout << "   -> parent[" << k << "] @" << p
                  << " (" << op_name(p->op) << ") receives grad shape "
                  << pshp.first << "x" << pshp.second << "\n";
    }
}

// --- VJP graph dump ---
void dump_vjp_dot(const Value& root, const std::string& filepath) {
    auto order = topo_from(root.node.get());
    std::ofstream out(filepath);
    if (!out) { std::cerr << "debug::dump_vjp_dot: failed to open " << filepath << "\n"; return; }

    auto shape_str = [](const Tensor& t){
        auto s = t.shape(); std::ostringstream os; os << s.first << "x" << s.second; return os.str();
    };

    out << "digraph VJP {\n"
           "  rankdir=LR;\n"
           "  node [shape=record, fontsize=10];\n";

    // forward nodes (same as dump_dot, but neutral colors)
    for (Node* n : order) {
        std::string color = (n->op==Op::Leaf ? (n->requires_grad ? "lightgoldenrod1" : "lightgrey")
                                             : (n->requires_grad ? "white" : "white"));
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
    auto shp = n->value.shape();
    std::cout << "[JVP] node @" << n
              << " op=" << op_name(n->op)
              << "  value=" << shp.first << "x" << shp.second << "\n";
    for (size_t k = 0; k < n->inputs.size(); ++k) {
        Node* p = n->inputs[k].get();
        auto pshp = p->value.shape();
        std::cout << "    parent[" << k << "] @" << p
                  << " (" << op_name(p->op) << ")  value="
                  << pshp.first << "x" << pshp.second << "\n";
    }
}

void dump_jvp_dot(const Value& root, const std::string& filepath) {
    auto order = topo_from(root.node.get());
    std::ofstream out(filepath);
    if (!out) { std::cerr << "debug::dump_jvp_dot: failed to open " << filepath << "\n"; return; }

    auto shape_str = [](const Tensor& t){
        auto s = t.shape(); std::ostringstream os; os << s.first << "x" << s.second; return os.str();
    };

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
