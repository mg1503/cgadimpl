#pragma once
#include <memory>
#include <string>
#include "ad/graph.hpp"

namespace ag::debug {

// ---- runtime controls ----
void enable_tracing(bool on = true);                    // print each node as it’s created
void set_print_limits(int max_rows, int max_cols,
                      int width = 10, int precision = 4);

// ---- printing utilities (call from tests if you want) ----
void print_tensor(const std::string& label, const Tensor& T);
void print_value (const std::string& label, const Value& v);
void print_grad  (const std::string& label, const Value& v);

// ---- whole-graph inspectors ----
void print_all_values(const Value& root);               // topo-ordered values
void print_all_grads (const Value& root);               // topo-ordered grads
void dump_dot(const Value& root, const std::string& filepath); // GraphViz .dot

// ---- internal: called by ops after creating a node ----
void on_node_created(const std::shared_ptr<Node>& n);

// ==========================================================================
// backprop implementation   --> Enable/disable live backprop step prints during backward()
// ==========================================================================

void enable_grad_tracing(bool on = true);

// Called from autodiff during backward() for each visited node
void on_backprop_step(Node* n, const Tensor& gy);

// Dump a DOT that shows the backward/VJP flow (red arrows child -> parent)
void dump_vjp_dot(const Value& root, const std::string& filepath);

// ==========================================================================
// jvp implementation
// ==========================================================================

void enable_jvp_tracing(bool on = true);           // turn on/off JVP step prints
void on_jvp_step(Node* n);                         // called once per node during jvp
void dump_jvp_dot(const Value& root, const std::string& filepath);  // parent->child (green)


} // namespace ag::debug
