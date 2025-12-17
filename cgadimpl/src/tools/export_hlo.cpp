// ========================================
// cgadimpl/src/tools/export_hlo.cpp
// ========================================
#include "ad/utils/export_hlo.hpp"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <iostream>
#include <limits>

namespace ag::hlo {
// --- 1. REWRITE THE HELPER FUNCTIONS ---

// New function to generate an N-D HLO type string
static std::string hlo_type_string(const Tensor& t) {
    std::ostringstream os;
    os << "tensor<";
    const auto& dims = t.shape().dims;
    if (dims.empty()) {
        os << "f32"; // Scalar case
    } else {
        for (size_t i = 0; i < dims.size(); ++i) {
            os << dims[i] << (i == dims.size() - 1 ? "" : "x");
        }
        os << "xf32>";
    }
    return os.str();
}

static std::string ty_scalar() {
    return "tensor<f32>";
}

// New N-D broadcast helper
static std::string maybe_broadcast(std::ostream& out,
                                   const std::string& vname,
                                   const Tensor& vT,
                                   const std::vector<int64_t>& target_dims,
                                   int& temp_id)
{
    const auto& src_dims = vT.shape().dims;
    if (src_dims == target_dims) {
        return vname;
    }

    // Build broadcast_dimensions for N-D sources to N-D targets.
    std::vector<int64_t> broadcast_dims;
    int src_rank = src_dims.size();
    int target_rank = target_dims.size();
    int rank_diff = target_rank - src_rank;

    for (int i = 0; i < src_rank; ++i) {
        // A dimension is part of the broadcast_dimensions attribute if it's NOT being broadcasted.
        // I.e., if its size is greater than 1.
        if (src_dims[i] > 1) {
            broadcast_dims.push_back(i + rank_diff);
        }
    }

    std::ostringstream dim_attr;
    dim_attr << "dense<[";
    for (size_t i = 0; i < broadcast_dims.size(); ++i) {
        if (i > 0) dim_attr << ", ";
        dim_attr << broadcast_dims[i];
    }
    dim_attr << "]> : tensor<" << broadcast_dims.size() << "xi64>";

    // --- THIS IS THE FIX ---
    // Manually construct the target type string instead of creating a temporary Tensor.
    std::stringstream target_type_ss;
    target_type_ss << "tensor<";
    if (target_dims.empty()) {
        target_type_ss << "f32";
    } else {
        for(size_t i = 0; i < target_dims.size(); ++i) {
            target_type_ss << target_dims[i] << (i == target_dims.size() - 1 ? "" : "x");
        }
        target_type_ss << "xf32>";
    }
    // --- END OF FIX ---

    std::string res = "%t" + std::to_string(temp_id++);
    out << "  " << res << " = stablehlo.broadcast_in_dim "
        << vname << ", broadcast_dimensions = " << dim_attr.str()
        << " : " << hlo_type_string(vT) << " -> " << target_type_ss.str() << "\n";
    
    return res;
}


void dump_stablehlo(const Value& root, const std::string& filepath)
{
    auto order = topo_from(root.node.get());

    // Number nodes to names (%argN for leaves; %vN otherwise)
    std::unordered_map<Node*, std::string> name;
    std::vector<Node*> args; args.reserve(order.size());

    // In this exporter, ALL leaves (Op::Leaf) become function arguments.
    for (Node* n : order) if (n->op == Op::Leaf) args.push_back(n);

    std::ofstream out(filepath);
    if (!out) {
        std::cerr << "export_hlo: failed to open " << filepath << "\n";
        return;
    }

    // Module header
    out << "module attributes { mhlo.dynamic_shape=\"false\" } {\n";

    // --- FIX #1: Use hlo_type_string and the correct way to access the root tensor ---
    // Function signature
    out << "  func.func @compute(";
    for (size_t i=0;i<args.size();++i) {
        Node* a = args[i];
        std::string an = "%arg" + std::to_string(i);
        name[a] = an;
        if (i) out << ", ";
        out << an << ": " << hlo_type_string(a->value); // This was already correct
    }
    // Use root.node->value to get the tensor
    out << ") -> " << hlo_type_string(root.node->value) << " {\n";

    int vid = 0;     // value id for %vN
    int tmpid = 0;   // temp id for broadcasts and constants

    // Helpers
    auto newv = [&](){ return std::string("%v") + std::to_string(vid++); };
    auto cst_scalar = [&](float v)->std::string{
        std::string cn = "%cst" + std::to_string(tmpid++);
        std::ostringstream lit;
        // The ty_scalar() helper is fine.
        lit << "  " << cn << " = stablehlo.constant dense<" << std::setprecision(8) << v
            << "> : " << ty_scalar() << "\n";
        out << lit.str();
        return cn;
    };
    
    // --- FIX #2: Update cst_zeros_like to use the new helper ---
    auto cst_zeros_like = [&](const Tensor& like)->std::string{
        auto cn = "%cst" + std::to_string(tmpid++);
        out << "  " << cn << " = stablehlo.constant dense<0.0> : " << hlo_type_string(like) << "\n";
        return cn;
    };

    // Emit each non-leaf in topo order
    for (Node* n : order) {
        if (n->op == Op::Leaf) continue;

        // --- THE FIX ---
        // 1. Get the full N-dimensional shape of the current node's output tensor.
        const auto& target_dims = n->value.shape().dims;

        // 2. The in_name helper now captures and uses this N-D shape vector.
        auto in_name = [&](size_t k)->std::string{
            Node* p = n->inputs[k].get();
            std::string pn = name.count(p) ? name[p] : (name[p]=newv()); // ensure parent has a name
            // Pass the full shape vector to our new maybe_broadcast function.
            return maybe_broadcast(out, pn, p->value, target_dims, tmpid);
        };

        switch (n->op) {
            // ----- Binary elementwise -----
            case Op::Add:
            case Op::Sub:
            case Op::Mul: {
                std::string a = in_name(0);
                std::string b = in_name(1);
                std::string v = newv();
                const char* op = (n->op==Op::Add) ? "add" : (n->op==Op::Sub) ? "subtract" : "multiply";
                out << "  " << v << " = stablehlo." << op << " " << a << ", " << b
                    << " : " << hlo_type_string(n->value) << "\n";
                name[n] = v;
                break;
            }

            // ----- Unary elementwise -----
            case Op::Relu: {
                std::string x = in_name(0);
                std::string zero = cst_scalar(0.0f);

                // --- THE FIX ---
                // 1. Create a temporary 1x1 tensor using the correct new constructor.
                Tensor one_by_one_tensor(Shape{{1, 1}}, /*requires_grad=*/false);
                
                // 2. Pass this temporary tensor to maybe_broadcast.
                //    Also, pass the full N-D target shape.
                const auto& target_dims = n->value.shape().dims;
                std::string z2 = maybe_broadcast(out, zero, one_by_one_tensor, target_dims, tmpid);
                // --- END OF FIX ---

                std::string v = newv();
                out << "  " << v << " = stablehlo.maximum " << x << ", " << z2
                    << " : " << hlo_type_string(n->value) << "\n"; // Use the N-D type helper
                name[n] = v;
                break;
            }

            case Op::Exp: case Op::Log: case Op::Tanh: {
                std::string x = in_name(0);
                std::string v = newv();
                const char* op = (n->op==Op::Exp) ? "exponential" : (n->op==Op::Log) ? "log" : "tanh";
                // This block was already correct as it used the new helpers.
                out << "  " << v << " = stablehlo." << op << " " << x
                    << " : " << hlo_type_string(n->value) << "\n";
                name[n] = v;
                break;
            }

            case Op::Sigmoid: {
                std::string x = in_name(0);
                std::string v = newv();
                // This was also correct.
                out << "  " << v << " = stablehlo.logistic " << x << " : " << hlo_type_string(n->value) << "\n";
                name[n] = v;
                break;
            }
            case Op::Softplus: {
                // softplus = log(1 + exp(x))
                std::string x = in_name(0);
                std::string ex = newv();
                out << "  " << ex << " = stablehlo.exponential " << x << " : " << hlo_type_string(n->value) << "\n";
                
                // --- FIX: Use the new N-D broadcast and correct Tensor constructor ---
                std::string one = cst_scalar(1.0f);
                Tensor one_by_one_tensor(Shape{{1, 1}}, false); // Correct constructor
                std::string oneb = maybe_broadcast(out, one, one_by_one_tensor, n->value.shape().dims, tmpid);
                // --- END FIX ---

                std::string add = newv();
                out << "  " << add << " = stablehlo.add " << ex << ", " << oneb << " : " << hlo_type_string(n->value) << "\n";
                std::string v = newv();
                out << "  " << v << " = stablehlo.log " << add << " : " << hlo_type_string(n->value) << "\n";
                name[n] = v;
                break;
            }
            case Op::SiLU: {
                // This block was already correct.
                std::string x = in_name(0);
                std::string s = newv();
                out << "  " << s << " = stablehlo.logistic " << x << " : " << hlo_type_string(n->value) << "\n";
                std::string v = newv();
                out << "  " << v << " = stablehlo.multiply " << x << ", " << s
                    << " : " << hlo_type_string(n->value) << "\n";
                name[n] = v;
                break;
            }
            case Op::GELU: {
                // --- FIX: This entire block needs to use the N-D broadcast ---
                std::string x = in_name(0);
                std::string x2 = newv(); out << "  " << x2 << " = stablehlo.multiply " << x << ", " << x << " : " << hlo_type_string(n->value) << "\n";
                std::string x3 = newv(); out << "  " << x3 << " = stablehlo.multiply " << x2 << ", " << x << " : " << hlo_type_string(n->value) << "\n";
                
                Tensor one_by_one_tensor(Shape{{1, 1}}, false); // Helper tensor
                const auto& target_dims = n->value.shape().dims;

                std::string c044 = cst_scalar(0.044715f);
                std::string c044b = maybe_broadcast(out, c044, one_by_one_tensor, target_dims, tmpid);
                std::string t1 = newv(); out << "  " << t1 << " = stablehlo.multiply " << x3 << ", " << c044b << " : " << hlo_type_string(n->value) << "\n";
                std::string t2 = newv(); out << "  " << t2 << " = stablehlo.add " << x << ", " << t1 << " : " << hlo_type_string(n->value) << "\n";
                
                std::string c = cst_scalar(0.797884583f); // sqrt(2/pi)
                std::string cb = maybe_broadcast(out, c, one_by_one_tensor, target_dims, tmpid);
                std::string u = newv(); out << "  " << u << " = stablehlo.multiply " << t2 << ", " << cb << " : " << hlo_type_string(n->value) << "\n";
                
                std::string th = newv(); out << "  " << th << " = stablehlo.tanh " << u << " : " << hlo_type_string(n->value) << "\n";
                
                std::string one = cst_scalar(1.0f);
                std::string oneb = maybe_broadcast(out, one, one_by_one_tensor, target_dims, tmpid);
                std::string s = newv(); out << "  " << s << " = stablehlo.add " << oneb << ", " << th << " : " << hlo_type_string(n->value) << "\n";
                
                std::string half = cst_scalar(0.5f);
                std::string halfb = maybe_broadcast(out, half, one_by_one_tensor, target_dims, tmpid);
                std::string h = newv(); out << "  " << h << " = stablehlo.multiply " << x << ", " << s << " : " << hlo_type_string(n->value) << "\n";
                
                std::string v = newv(); out << "  " << v << " = stablehlo.multiply " << h << ", " << halfb << " : " << hlo_type_string(n->value) << "\n";
                name[n] = v;
                break;
            }
            case Op::LeakyRelu: {
                // --- FIX: Use N-D broadcast and correct API calls ---
                std::string x = in_name(0);
                Node* A = n->inputs[1].get();
                float alpha = A->value.data<float>()[0]; // Correct way to get scalar value
                
                Tensor one_by_one_tensor(Shape{{1, 1}}, false);
                const auto& target_dims = n->value.shape().dims;

                std::string a = cst_scalar(alpha);
                std::string ab = maybe_broadcast(out, a, one_by_one_tensor, target_dims, tmpid);
                std::string ax = newv(); out << "  " << ax << " = stablehlo.multiply " << ab << ", " << x << " : " << hlo_type_string(n->value) << "\n";
                
                std::string zero = cst_scalar(0.0f);
                std::string z2 = maybe_broadcast(out, zero, one_by_one_tensor, target_dims, tmpid);
                
                std::string pred = newv();
                out << "  " << pred << " = stablehlo.compare GT, " << x << ", " << z2
                    << ", type = \"PRED\" : " << hlo_type_string(n->value) << "\n";
                
                std::string v = newv();
                out << "  " << v << " = stablehlo.select " << pred << ", " << x << ", " << ax
                    << " : tensor<" << target_dims[0] << "x" << target_dims[1] << "xi1>, " 
                    << hlo_type_string(n->value) << ", " << hlo_type_string(n->value) 
                    << " -> " << hlo_type_string(n->value) << "\n";
                name[n] = v;
                break;
            }

            // ----- Matmul -----
            case Op::MatMul: {
                // 2D matmul: dot_general with contracting dims lhs[1], rhs[0]
                Node* A = n->inputs[0].get();
                Node* B = n->inputs[1].get();
                std::string an = name.count(A) ? name[A] : (name[A]=newv());
                std::string bn = name.count(B) ? name[B] : (name[B]=newv());
                std::string v = newv();
                out << "  " << v << " = stablehlo.dot_general " << an << ", " << bn
                    << " contracting_dims = [lhs = [1], rhs = [0]], "
                       "batching_dims = [lhs = [], rhs = []] "
                    << " : " << hlo_type_string(A->value) << ", " << hlo_type_string(B->value)
                    << " -> " << hlo_type_string(n->value) << "\n";
                name[n] = v;
                break;
            }

            // ----- Reductions -----
            case Op::Sum: {
                // Reduce over both dims with add; init 0
                Node* X = n->inputs[0].get();
                std::string xn = name.count(X) ? name[X] : (name[X]=newv());
                std::string zero = cst_scalar(0.0f);
                std::string v = newv();
                // For readability (and many verifiers accept this), we put a simple scalar region.
                out << "  " << v << " = stablehlo.reduce " << xn << ", " << zero
                    << " dimensions = [0, 1] : "
                    << hlo_type_string(X->value) << ", " << ty_scalar() << " -> " << ty_scalar() << " {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32):\n"
                    << "      %r = stablehlo.add %lhs, %rhs : f32\n"
                    << "      stablehlo.return %r : f32\n"
                    << "  }\n";
                name[n] = v;
                break;
            }
            case Op::RowSum: {
                // --- FIX: Use N-D API ---
                Node* X = n->inputs[0].get();
                std::string xn = name.count(X) ? name[X] : (name[X]=newv());
                std::string zero = cst_scalar(0.0f);
                std::string v = newv();
                out << "  " << v << " = stablehlo.reduce " << xn << ", " << zero
                    << " dimensions = dense<[1]> : tensor<1xi64>" // reduce axis 1
                    << " : (" << hlo_type_string(X->value) << ", tensor<f32>"
                    << ") -> " << hlo_type_string(n->value) << " {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32):\n"
                    << "      %r = stablehlo.add %lhs, %rhs : f32\n"
                    << "      stablehlo.return %r : f32\n"
                    << "  }\n";
                name[n] = v;
                break;
            }
            case Op::RowMax: {
                Node* X = n->inputs[0].get();
                std::string xn = name.count(X) ? name[X] : (name[X]=newv());
                std::string ninf = cst_scalar(-std::numeric_limits<float>::infinity());
                std::string v = newv();
                
                // --- FIX: Use N-D hlo_type_string for the output type ---
                out << "  " << v << " = stablehlo.reduce " << xn << ", " << ninf
                    << " dimensions = dense<[1]> : tensor<1xi64>" // reduce axis 1
                    << " : (" << hlo_type_string(X->value) << ", tensor<f32>"
                    << ") -> " << hlo_type_string(n->value) << " {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32):\n"
                    << "      %r = stablehlo.maximum %lhs, %rhs : f32\n"
                    << "      stablehlo.return %r : f32\n"
                    << "  }\n";
                name[n] = v;
                break;
            }
            case Op::MeanAll: {
                Node* X = n->inputs[0].get();
                // --- FIX: Use .numel() for correctness with N-D tensors ---
                std::string ones = cst_scalar(1.0f / static_cast<float>(X->value.numel()));
                
                std::string xn = name.count(X) ? name[X] : (name[X]=newv());
                std::string zero = cst_scalar(0.0f);
                std::string s = newv();

                // --- FIX: Build the dimensions attribute dynamically for N-D ---
                std::vector<int64_t> all_dims(X->value.shape().dims.size());
                std::iota(all_dims.begin(), all_dims.end(), 0); // Fills with 0, 1, 2, ...
                
                std::ostringstream dims_attr;
                dims_attr << "dense<[";
                for(size_t i=0; i<all_dims.size(); ++i) { dims_attr << (i > 0 ? ", " : "") << all_dims[i]; }
                dims_attr << "]> : tensor<" << all_dims.size() << "xi64>";
                // --- END FIX ---

                out << "  " << s << " = stablehlo.reduce " << xn << ", " << zero
                    << " dimensions = " << dims_attr.str() << " : "
                    << "(" << hlo_type_string(X->value) << ", tensor<f32>"
                    << ") -> " << ty_scalar() << " {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32):\n"
                    << "      %r = stablehlo.add %lhs, %rhs : f32\n"
                    << "      stablehlo.return %r : f32\n"
                    << "  }\n";

                std::string v = newv();
                out << "  " << v << " = stablehlo.multiply " << s << ", " << ones
                    << " : " << ty_scalar() << "\n";
                name[n] = v;
                break;
            }

            // ----- Softmax / LogSumExp -----
            case Op::SoftmaxRow:
            case Op::LogSumExpRow: {
                Node* Z = n->inputs[0].get();
                std::string zn = name.count(Z) ? name[Z] : (name[Z]=newv());

                // --- FIX: Get the correct shape for the intermediate tensor 'm' ---
                // The output of the row-max/row-sum is the same shape as the final output `n`.
                const std::string m_type = hlo_type_string(n->value);

                // m = row_max(z)
                std::string ninf = cst_scalar(-std::numeric_limits<float>::infinity());
                std::string m = newv();
                out << "  " << m << " = stablehlo.reduce " << zn << ", " << ninf
                    << " dimensions = dense<[1]> : tensor<1xi64>"
                    << " : (" << hlo_type_string(Z->value) << ", tensor<f32>"
                    << ") -> " << m_type << " {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32):\n"
                    << "      %r = stablehlo.maximum %lhs, %rhs : f32\n"
                    << "      stablehlo.return %r : f32\n"
                    << "  }\n";
                
                // zc = z - m (broadcasting m)
                std::string mb = newv();
                out << "  " << mb << " = stablehlo.broadcast_in_dim " << m
                    << ", broadcast_dimensions = dense<[0]> : tensor<1xi64>"
                    << " : " << m_type << " -> " << hlo_type_string(Z->value) << "\n";
                
                std::string zc = newv();
                out << "  " << zc << " = stablehlo.subtract " << zn << ", " << mb
                    << " : " << hlo_type_string(Z->value) << "\n";
                
                // e = exp(zc)
                std::string e = newv(); 
                out << "  " << e << " = stablehlo.exponential " << zc << " : " << hlo_type_string(Z->value) << "\n";
                
                // s = row_sum(e)
                std::string zero = cst_scalar(0.0f);
                std::string s = newv();
                out << "  " << s << " = stablehlo.reduce " << e << ", " << zero
                    << " dimensions = dense<[1]> : tensor<1xi64>"
                    << " : (" << hlo_type_string(Z->value) << ", tensor<f32>"
                    << ") -> " << m_type << " {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32):\n"
                    << "      %r = stablehlo.add %lhs, %rhs : f32\n"
                    << "      stablehlo.return %r : f32\n"
                    << "  }\n";

                if (n->op == Op::LogSumExpRow) {
                    // lse = log(s) + m
                    std::string ls = newv(); 
                    out << "  " << ls << " = stablehlo.log " << s << " : " << m_type << "\n";
                    std::string v = newv(); 
                    out << "  " << v << " = stablehlo.add " << ls << ", " << m << " : " << m_type << "\n";
                    name[n] = v;
                } else { // SoftmaxRow
                    // p = e / s
                    std::string sb = newv();
                    out << "  " << sb << " = stablehlo.broadcast_in_dim " << s
                        << ", broadcast_dimensions = dense<[0]> : tensor<1xi64>"
                        << " : " << m_type << " -> " << hlo_type_string(Z->value) << "\n";
                    std::string v = newv();
                    out << "  " << v << " = stablehlo.divide " << e << ", " << sb
                        << " : " << hlo_type_string(Z->value) << "\n";
                    name[n] = v;
                }
                break;
            }

            // ----- Loss -----
            case Op::CeWithLogits: {
                // Formula: CE = -mean( sum( Y * log_softmax(Z) , axis=-1) )
                Node* Z = n->inputs[0].get();
                Node* Y = n->inputs[1].get();
                std::string zn = name.count(Z) ? name[Z] : (name[Z]=newv());
                std::string yn = name.count(Y) ? name[Y] : (name[Y]=newv());

                // --- FIX: Use N-D shapes and helpers ---
                const auto& z_dims = Z->value.shape().dims;
                const auto& y_dims = Y->value.shape().dims;
                int64_t B = z_dims.empty() ? 1 : z_dims[0]; // Batch size
                
                // We need the shape of the row-wise reduction result (e.g., [B, 1])
                std::vector<int64_t> reduced_shape_dims = z_dims;
                if (!reduced_shape_dims.empty()) {
                    reduced_shape_dims.back() = 1;
                }
                Tensor reduced_shape_tensor(Shape{reduced_shape_dims}, false);
                const std::string reduced_type_str = hlo_type_string(reduced_shape_tensor);
                // --- END FIX ---


                // --- Stable log_softmax(Z) implementation (same as LogSumExpRow) ---
                std::string ninf = cst_scalar(-std::numeric_limits<float>::infinity());
                std::string m = newv();
                out << "  " << m << " = stablehlo.reduce " << zn << ", " << ninf
                    << " dimensions = dense<[" << z_dims.size() - 1 << "]> : tensor<1xi64>"
                    << " : (" << hlo_type_string(Z->value) << ", tensor<f32>"
                    << ") -> " << reduced_type_str << " {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32): %r = stablehlo.maximum %lhs, %rhs : f32\n      stablehlo.return %r : f32\n  }\n";
                
                std::string mb = newv();
                out << "  " << mb << " = stablehlo.broadcast_in_dim " << m
                    << ", broadcast_dimensions = dense<[0]> : tensor<1xi64>" // Assuming 2D for simplicity of broadcast
                    << " : " << reduced_type_str << " -> " << hlo_type_string(Z->value) << "\n";
                
                std::string zc = newv(); out << "  " << zc << " = stablehlo.subtract " << zn << ", " << mb << " : " << hlo_type_string(Z->value) << "\n";
                std::string e = newv();  out << "  " << e  << " = stablehlo.exponential " << zc << " : " << hlo_type_string(Z->value) << "\n";
                std::string zero = cst_scalar(0.0f);
                std::string s = newv();
                out << "  " << s << " = stablehlo.reduce " << e << ", " << zero
                    << " dimensions = dense<[" << z_dims.size() - 1 << "]> : tensor<1xi64>"
                    << " : (" << hlo_type_string(Z->value) << ", tensor<f32>"
                    << ") -> " << reduced_type_str << " {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32): %r = stablehlo.add %lhs, %rhs : f32\n      stablehlo.return %r : f32\n  }\n";
                
                std::string lse = newv(); out << "  " << lse << " = stablehlo.log " << s << " : " << reduced_type_str << "\n";
                std::string lse_plus_m = newv(); out << "  " << lse_plus_m << " = stablehlo.add " << lse << ", " << m << " : " << reduced_type_str << "\n";
                
                // log_softmax = Z - (lse + m)
                std::string lse_b = newv();
                out << "  " << lse_b << " = stablehlo.broadcast_in_dim " << lse_plus_m
                    << ", broadcast_dimensions = dense<[0]> : tensor<1xi64>"
                    << " : " << reduced_type_str << " -> " << hlo_type_string(Z->value) << "\n";
                std::string lsm = newv(); out << "  " << lsm << " = stablehlo.subtract " << zn << ", " << lse_b << " : " << hlo_type_string(Z->value) << "\n";
                
                // prod = Y * log_softmax
                std::string prod = newv(); out << "  " << prod << " = stablehlo.multiply " << yn << ", " << lsm << " : " << hlo_type_string(Z->value) << "\n";

                // rs = sum(prod, axis=-1)
                std::string rs = newv();
                out << "  " << rs << " = stablehlo.reduce " << prod << ", " << zero
                    << " dimensions = dense<[" << z_dims.size() - 1 << "]> : tensor<1xi64>"
                    << " : (" << hlo_type_string(Z->value) << ", tensor<f32>"
                    << ") -> " << reduced_type_str << " {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32): %r = stablehlo.add %lhs, %rhs : f32\n      stablehlo.return %r : f32\n  }\n";

                // mean over batch and negate
                std::string sB = cst_scalar(-1.0f / static_cast<float>(B));
                std::string ssum = newv(); // sum over batch (dim 0)
                out << "  " << ssum << " = stablehlo.reduce " << rs << ", " << zero
                    << " dimensions = dense<[0]> : tensor<1xi64>"
                    << " : (" << reduced_type_str << ", tensor<f32>"
                    << ") -> " << ty_scalar() << " {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32): %r = stablehlo.add %lhs, %rhs : f32\n      stablehlo.return %r : f32\n  }\n";
                
                std::string v = newv();
                out << "  " << v << " = stablehlo.multiply " << ssum << ", " << sB
                    << " : " << ty_scalar() << "\n";
                name[n] = v;
                break;
            }

            default: {
                // Fallback: identity (shouldn't happen)
                Node* X = n->inputs.empty() ? nullptr : n->inputs[0].get();
                std::string xn = X ? (name.count(X)?name[X]:(name[X]=newv())) : "%UNDEF";
                std::string v = newv();
                // This was already corrected.
                out << "  " << v << " = stablehlo.copy " << xn << " : " << (X ? hlo_type_string(X->value) : "tensor<?>") << "\n";
                name[n] = v;
                break;
            }
        } // switch
    }

    // Return root
    std::string ret = name[root.node.get()];
    if (ret.empty()) ret = "%UNDEF";
    out << "  return " << ret << " : " << hlo_type_string(root.val()) << "\n";
    out << "  }\n"; // func
    out << "}\n";    // module

    out.close();
    std::cout << "Wrote StableHLO MLIR to: " << filepath << "\n";
}

} // namespace ag::hlo
