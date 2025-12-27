// =========================================
// cgadimpl/src/tools/debug.cpp
// =========================================
#include "ad/utils/debug.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <chrono>
#include <functional>
#include "ad/autodiff/autodiff.hpp"

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

// dtype extractor
std::string dtype_str( const Tensor& t){
    std::ostringstream os;

    std::string dtype_var = get_dtype_name(t.dtype());
    os<<dtype_var;
    return os.str();
}


// Helper to handle automatic rendering if filename ends in .png, .jpg, etc.
void auto_render(const std::string& dot_path, const std::string& target_path) {
    if (dot_path == target_path) return; // Nothing to do if it's already a .dot file

    std::string ext = "";
    size_t dot_pos = target_path.find_last_of('.');
    if (dot_pos != std::string::npos) {
        ext = target_path.substr(dot_pos + 1);
    }

    // Supported image formats for Graphviz
    if (ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "pdf" || ext == "svg") {
        std::string cmd = "dot -T" + ext + " " + dot_path + " -o " + target_path;
        int ret = std::system(cmd.c_str());
        if (ret == 0) {
            std::cout << "Successfully rendered image to: " << target_path << "\n";
            // Optionally remove the temporary .dot file
            std::string rm_cmd = "rm " + dot_path;
            std::system(rm_cmd.c_str());
        } else {
            std::cerr << "Error: 'dot' command failed to render image. DOT file preserved at: " << dot_path << "\n";
        }
    }
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
        lab  << op_name(n->op) << "\\n" << shape_str(n->value) << "\\n"<< dtype_str(n->value) ; // Correct
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

    // If the user provided an image extension, render it.
    std::string dot_file = filepath;
    bool is_image = false;
    if (filepath.size() > 4) {
        std::string ext = filepath.substr(filepath.size() - 4);
        if (ext == ".png" || ext == ".jpg" || ext == ".svg" || ext == ".pdf") is_image = true;
    }
    if (filepath.size() > 5 && filepath.substr(filepath.size() - 5) == ".jpeg") is_image = true;

    if (is_image) {
        dot_file = filepath + ".dot";
        // Rename the original file to .dot temporarily
        std::rename(filepath.c_str(), dot_file.c_str());
        auto_render(dot_file, filepath);
    } else {
        std::cout << "Wrote graph DOT to: " << filepath << "\n"
                  "Render with: dot -Tpng " << filepath << " -o build/graph.png\n";    
    }

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

    std::string dot_file = filepath;
    bool is_image = false;
    if (filepath.size() > 4) {
        std::string ext = filepath.substr(filepath.size() - 4);
        if (ext == ".png" || ext == ".jpg" || ext == ".svg" || ext == ".pdf") is_image = true;
    }
    if (filepath.size() > 5 && filepath.substr(filepath.size() - 5) == ".jpeg") is_image = true;

    if (is_image) {
        dot_file = filepath + ".dot";
        std::rename(filepath.c_str(), dot_file.c_str());
        auto_render(dot_file, filepath);
    } else {
        std::cout << "Wrote VJP DOT to: " << filepath << "\n"
                     "Render with: dot -Tpng " << filepath << " -o build/graph_vjp.png\n";
    }
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

    std::string dot_file = filepath;
    bool is_image = false;
    if (filepath.size() > 4) {
        std::string ext = filepath.substr(filepath.size() - 4);
        if (ext == ".png" || ext == ".jpg" || ext == ".svg" || ext == ".pdf") is_image = true;
    }
    if (filepath.size() > 5 && filepath.substr(filepath.size() - 5) == ".jpeg") is_image = true;

    if (is_image) {
        dot_file = filepath + ".dot";
        std::rename(filepath.c_str(), dot_file.c_str());
        auto_render(dot_file, filepath);
    } else {
        std::cout << "Wrote JVP DOT to: " << filepath
                  << "\nRender: dot -Tpng " << filepath << " -o build/graph_jvp.png\n";
    }
}

void print_dag_summary(const Value& root) {
    if (!root.node) {
        std::cout << "DAG Summary: Empty graph\n";
        return;
    }
    auto order = topo_from(root.node.get());
    size_t num_nodes = order.size();
    size_t num_leaves = 0;
    size_t num_trainable = 0;
    size_t total_params = 0;
    std::unordered_map<Op, int> op_counts;

    for (Node* n : order) {
        op_counts[n->op]++;
        if (n->is_leaf) {
            num_leaves++;
            if (n->requires_grad()) {
                num_trainable++;
                total_params += n->value.numel();
            }
        }
    }

    std::cout << "=== DAG Summary ===\n";
    std::cout << "Total Nodes:      " << num_nodes << "\n";
    std::cout << "Leaf Nodes:       " << num_leaves << " (" << num_trainable << " trainable)\n";
    std::cout << "Trainable Params: " << total_params << "\n";
    std::cout << "Op Breakdown:\n";
    for (auto const& [op, count] : op_counts) {
        std::cout << "  - " << std::left << std::setw(15) << op_name(op) << ": " << count << "\n";
    }
    std::cout << "===================\n";
}

bool validate_dag(const Value& root) {
    if (!root.node) return true;
    
    // Cycle detection using DFS (White/Gray/Black coloring)
    std::unordered_map<Node*, int> color; // 0=White, 1=Gray, 2=Black
    
    std::function<bool(Node*)> has_cycle = [&](Node* n) -> bool {
        color[n] = 1; // Gray
        for (auto& input : n->inputs) {
            if (!input) continue;
            if (color[input.get()] == 1) return true; // Found gray node -> cycle
            if (color[input.get()] == 0) {
                if (has_cycle(input.get())) return true;
            }
        }
        color[n] = 2; // Black
        return false;
    };

    if (has_cycle(root.node.get())) {
        std::cerr << "DAG Validation Failed: Cycle detected in graph!\n";
        return false;
    }

    // Check connectivity (all nodes path to root) - inherent in topo_from/traversal
    // Check for null inputs
    auto order = topo_from(root.node.get());
    for (Node* n : order) {
        for (size_t i = 0; i < n->inputs.size(); ++i) {
            if (!n->inputs[i]) {
                std::cerr << "DAG Validation Warning: Node " << op_name(n->op) << " @" << n 
                          << " has null input at index " << i << "\n";
            }
        }
    }

    std::cout << "DAG Validation successful: No cycles found, " << order.size() << " nodes reachable.\n";
    return true;
}

void benchmark_dag(const Value& root, int iterations) {
    if (!root.node) return;
    
    std::cout << "=== Starting DAG Benchmark (" << iterations << " iterations) ===\n";
    
    // 1. Measure Forward (approx by topo + dummy traverse if we don't have a clean 'forward' handle)
    // Actually, 'root' is already the result of forward. 
    // To bench forward properly, we'd need a closure. 
    // Let's bench topo vs total.
    
    auto start_topo = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        auto order = topo_from(root.node.get());
    }
    auto end_topo = std::chrono::high_resolution_clock::now();
    
    // 2. Measure Backward
    auto start_bw = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        backward(root, nullptr, false); // sequential
    }
    auto end_bw = std::chrono::high_resolution_clock::now();
    
    auto start_bw_p = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        backward(root, nullptr, true); // parallel
    }
    auto end_bw_p = std::chrono::high_resolution_clock::now();
    
    auto topo_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_topo - start_topo).count() / (float)iterations;
    auto bw_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_bw - start_bw).count() / (float)iterations;
    auto bw_p_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_bw_p - start_bw_p).count() / (float)iterations;
    
    std::cout << "Topo Sort:        " << std::fixed << std::setprecision(2) << topo_ms << " us\n";
    std::cout << "Backward (seq):   " << bw_ms << " us\n";
    std::cout << "Backward (par):   " << bw_p_ms << " us\n";
    if (bw_ms > 0)
        std::cout << "Parallel Speedup: " << (float)bw_ms / bw_p_ms << "x\n";
    std::cout << "===========================================\n";
}

} // namespace ag::debug
