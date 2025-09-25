#include "ad/export_hlo.hpp"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <iostream>
#include <limits>

namespace ag::hlo {

static std::string ty2d(const Tensor& t) {
    auto [r,c] = t.shape();
    std::ostringstream os;
    os << "tensor<" << r << "x" << c << "xf32>";
    return os.str();
}

static std::string ty_scalar() {
    return "tensor<f32>";
}

// Decide if we must broadcast A to match outShape (R,C); if so emit broadcast op and return name.
// If `dims` is inferred for rank-2 only: [ (r>1?0: - ), (c>1?1: - ) ]
static std::string maybe_broadcast(std::ostream& out,
                                   const std::string& vname,
                                   const Tensor& vT,
                                   int R, int C,
                                   int& temp_id)
{
    int r=vT.rows(), c=vT.cols();
    if (r==R && c==C) return vname;

    // Build broadcast_dimensions for rank-2 sources to rank-2 targets.
    // If source dim is 1, it's broadcasted (not listed); else mapped to same index.
    std::vector<int> dims;
    if (r>1) dims.push_back(0);
    if (c>1) dims.push_back(1);

    std::ostringstream dim_attr;
    dim_attr << "dense<[";
    for (size_t i=0;i<dims.size();++i) { if (i) dim_attr << ", "; dim_attr << dims[i]; }
    dim_attr << "]> : tensor<" << dims.size() << "xi64>";

    std::string res = "%t" + std::to_string(temp_id++);
    out << "  " << res << " = stablehlo.broadcast_in_dim "
        << vname << ", broadcast_dimensions = " << dim_attr.str()
        << " : " << ty2d(vT) << " -> tensor<" << R << "x" << C << "xf32>\n";
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

    // Function signature
    out << "  func.func @compute(";
    for (size_t i=0;i<args.size();++i) {
        Node* a = args[i];
        std::string an = "%arg" + std::to_string(i);
        name[a] = an;
        if (i) out << ", ";
        out << an << ": " << ty2d(a->value);
    }
    out << ") -> " << ty2d(root.val()) << " {\n";

    int vid = 0;     // value id for %vN
    int tmpid = 0;   // temp id for broadcasts and constants

    // Helpers
    auto newv = [&](){ return std::string("%v") + std::to_string(vid++); };
    auto cst_scalar = [&](float v)->std::string{
        std::string cn = "%cst" + std::to_string(tmpid++);
        // StableHLO prefers constant tensors; we emit rank-0 tensor<f32>
        std::ostringstream lit;
        // Use mlir dense format for scalars: dense<value> : tensor<f32>
        lit << "  " << cn << " = stablehlo.constant dense<" << std::setprecision(8) << v
            << "> : " << ty_scalar() << "\n";
        out << lit.str();
        return cn;
    };
    auto cst_zeros_like = [&](const Tensor& like)->std::string{
        auto cn = "%cst" + std::to_string(tmpid++);
        out << "  " << cn << " = stablehlo.constant dense<0.0> : " << ty2d(like) << "\n";
        return cn;
    };

    // Emit each non-leaf in topo order
    for (Node* n : order) {
        if (n->op == Op::Leaf) continue;

        // Fetch (and broadcast if necessary) inputs
        auto R = n->value.rows(), C = n->value.cols();
        auto in_name = [&](size_t k)->std::string{
            Node* p = n->inputs[k].get();
            std::string pn = name.count(p) ? name[p] : (name[p]=newv()); // ensure parent has a name
            return maybe_broadcast(out, pn, p->value, R, C, tmpid);
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
                    << " : " << ty2d(n->value) << "\n";
                name[n] = v;
                break;
            }

            // ----- Unary elementwise -----
            case Op::Relu: {
                std::string x = in_name(0);
                std::string zero = cst_scalar(0.0f);
                std::string z2 = maybe_broadcast(out, zero, Tensor(1,1), R, C, tmpid);
                std::string v = newv();
                out << "  " << v << " = stablehlo.maximum " << x << ", " << z2
                    << " : " << ty2d(n->value) << "\n";
                name[n] = v;
                break;
            }
            case Op::Exp: case Op::Log: case Op::Tanh: {
                std::string x = in_name(0);
                std::string v = newv();
                const char* op = (n->op==Op::Exp) ? "exponential" : (n->op==Op::Log) ? "log" : "tanh";
                out << "  " << v << " = stablehlo." << op << " " << x
                    << " : " << ty2d(n->value) << "\n";
                name[n] = v;
                break;
            }
            case Op::Sigmoid: {
                // sigmoid(x) = logistic(x) exists in StableHLO
                std::string x = in_name(0);
                std::string v = newv();
                out << "  " << v << " = stablehlo.logistic " << x << " : " << ty2d(n->value) << "\n";
                name[n] = v;
                break;
            }
            case Op::Softplus: {
                // softplus = log(1 + exp(x))
                std::string x = in_name(0);
                std::string ex = newv();
                out << "  " << ex << " = stablehlo.exponential " << x << " : " << ty2d(n->value) << "\n";
                std::string one = cst_scalar(1.0f);
                std::string oneb = maybe_broadcast(out, one, Tensor(1,1), R, C, tmpid);
                std::string add = newv();
                out << "  " << add << " = stablehlo.add " << ex << ", " << oneb << " : " << ty2d(n->value) << "\n";
                std::string v = newv();
                out << "  " << v << " = stablehlo.log " << add << " : " << ty2d(n->value) << "\n";
                name[n] = v;
                break;
            }
            case Op::SiLU: {
                // silu = x * sigmoid(x)
                std::string x = in_name(0);
                std::string s = newv();
                out << "  " << s << " = stablehlo.logistic " << x << " : " << ty2d(n->value) << "\n";
                std::string v = newv();
                out << "  " << v << " = stablehlo.multiply " << x << ", " << s
                    << " : " << ty2d(n->value) << "\n";
                name[n] = v;
                break;
            }
            case Op::GELU: {
                // tanh-approx GELU
                // u = sqrt(2/pi)*(x + 0.044715 x^3); gelu = 0.5*x*(1+tanh(u))
                std::string x = in_name(0);
                std::string x2 = newv(); out << "  " << x2 << " = stablehlo.multiply " << x << ", " << x << " : " << ty2d(n->value) << "\n";
                std::string x3 = newv(); out << "  " << x3 << " = stablehlo.multiply " << x2 << ", " << x << " : " << ty2d(n->value) << "\n";
                std::string c044 = cst_scalar(0.044715f);
                std::string c044b = maybe_broadcast(out, c044, Tensor(1,1), R, C, tmpid);
                std::string t1 = newv(); out << "  " << t1 << " = stablehlo.multiply " << x3 << ", " << c044b << " : " << ty2d(n->value) << "\n";
                std::string t2 = newv(); out << "  " << t2 << " = stablehlo.add " << x << ", " << t1 << " : " << ty2d(n->value) << "\n";
                std::string c = cst_scalar(0.797884583f); // sqrt(2/pi)
                std::string cb = maybe_broadcast(out, c, Tensor(1,1), R, C, tmpid);
                std::string u = newv(); out << "  " << u << " = stablehlo.multiply " << t2 << ", " << cb << " : " << ty2d(n->value) << "\n";
                std::string th = newv(); out << "  " << th << " = stablehlo.tanh " << u << " : " << ty2d(n->value) << "\n";
                std::string one = cst_scalar(1.0f);
                std::string oneb = maybe_broadcast(out, one, Tensor(1,1), R, C, tmpid);
                std::string s = newv(); out << "  " << s << " = stablehlo.add " << oneb << ", " << th << " : " << ty2d(n->value) << "\n";
                std::string half = cst_scalar(0.5f);
                std::string halfb = maybe_broadcast(out, half, Tensor(1,1), R, C, tmpid);
                std::string h = newv(); out << "  " << h << " = stablehlo.multiply " << x << ", " << s << " : " << ty2d(n->value) << "\n";
                std::string v = newv(); out << "  " << v << " = stablehlo.multiply " << h << ", " << halfb << " : " << ty2d(n->value) << "\n";
                name[n] = v;
                break;
            }
            case Op::LeakyRelu: {
                // select(x > 0, x, alpha*x)
                std::string x = in_name(0);
                Node* A = n->inputs[1].get();
                float alpha = A->value(0,0);
                std::string a = cst_scalar(alpha);
                std::string ab = maybe_broadcast(out, a, Tensor(1,1), R, C, tmpid);
                std::string ax = newv(); out << "  " << ax << " = stablehlo.multiply " << ab << ", " << x << " : " << ty2d(n->value) << "\n";
                std::string zero = cst_scalar(0.0f);
                std::string z2 = maybe_broadcast(out, zero, Tensor(1,1), R, C, tmpid);
                std::string pred = newv();
                out << "  " << pred << " = stablehlo.compare GT " << x << ", " << z2
                    << " : " << ty2d(n->value) << "\n";
                std::string v = newv();
                out << "  " << v << " = stablehlo.select " << pred << ", " << x << ", " << ax
                    << " : " << ty2d(n->value) << "\n";
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
                    << " : " << ty2d(A->value) << ", " << ty2d(B->value)
                    << " -> " << ty2d(n->value) << "\n";
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
                    << ty2d(X->value) << ", " << ty_scalar() << " -> " << ty_scalar() << " {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32):\n"
                    << "      %r = stablehlo.add %lhs, %rhs : f32\n"
                    << "      stablehlo.return %r : f32\n"
                    << "  }\n";
                name[n] = v;
                break;
            }
            case Op::RowSum: {
                Node* X = n->inputs[0].get();
                std::string xn = name.count(X) ? name[X] : (name[X]=newv());
                std::string zero = cst_scalar(0.0f);
                std::string v = newv();
                out << "  " << v << " = stablehlo.reduce " << xn << ", " << zero
                    << " dimensions = [1] : "
                    << ty2d(X->value) << ", " << ty_scalar()
                    << " -> tensor<" << X->value.rows() << "x1xf32> {\n"
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
                // init = -inf
                std::string ninf = cst_scalar(-std::numeric_limits<float>::infinity());
                std::string v = newv();
                out << "  " << v << " = stablehlo.reduce " << xn << ", " << ninf
                    << " dimensions = [1] : "
                    << ty2d(X->value) << ", " << ty_scalar()
                    << " -> tensor<" << X->value.rows() << "x1xf32> {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32):\n"
                    << "      %r = stablehlo.maximum %lhs, %rhs : f32\n"
                    << "      stablehlo.return %r : f32\n"
                    << "  }\n";
                name[n] = v;
                break;
            }
            case Op::MeanAll: {
                Node* X = n->inputs[0].get();
                // mean = sum / (R*C)
                std::string ones = cst_scalar(1.0f / float(X->value.rows()*X->value.cols()));
                // Reuse Sum lowering then multiply
                // Emit sum node on the fly
                std::string xn = name.count(X) ? name[X] : (name[X]=newv());
                std::string zero = cst_scalar(0.0f);
                std::string s = newv();
                out << "  " << s << " = stablehlo.reduce " << xn << ", " << zero
                    << " dimensions = [0, 1] : "
                    << ty2d(X->value) << ", " << ty_scalar() << " -> " << ty_scalar() << " {\n"
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
            case Op::SoftmaxRow: {
                Node* Z = n->inputs[0].get();
                std::string zn = name.count(Z) ? name[Z] : (name[Z]=newv());
                int B = Z->value.rows(), C = Z->value.cols();
                // m = row_max(z)
                std::string ninf = cst_scalar(-std::numeric_limits<float>::infinity());
                std::string m = newv();
                out << "  " << m << " = stablehlo.reduce " << zn << ", " << ninf
                    << " dimensions = [1] : " << ty2d(Z->value) << ", " << ty_scalar()
                    << " -> tensor<" << B << "x1xf32> {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32):\n"
                    << "      %r = stablehlo.maximum %lhs, %rhs : f32\n"
                    << "      stablehlo.return %r : f32\n"
                    << "  }\n";
                // zc = z - m
                std::string mb = "%mb" + std::to_string(tmpid++);
                out << "  " << mb << " = stablehlo.broadcast_in_dim " << m
                    << ", broadcast_dimensions = dense<[0]> : tensor<" << B << "x1xf32> -> "
                    << ty2d(Z->value) << "\n";
                std::string zc = newv();
                out << "  " << zc << " = stablehlo.subtract " << zn << ", " << mb
                    << " : " << ty2d(Z->value) << "\n";
                // e = exp(zc)
                std::string e = newv(); out << "  " << e << " = stablehlo.exponential " << zc << " : " << ty2d(Z->value) << "\n";
                // s = row_sum(e)
                std::string zero = cst_scalar(0.0f);
                std::string s = newv();
                out << "  " << s << " = stablehlo.reduce " << e << ", " << zero
                    << " dimensions = [1] : " << ty2d(Z->value) << ", " << ty_scalar()
                    << " -> tensor<" << B << "x1xf32> {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32):\n"
                    << "      %r = stablehlo.add %lhs, %rhs : f32\n"
                    << "      stablehlo.return %r : f32\n"
                    << "  }\n";
                // p = e / s
                std::string sb = "%sb" + std::to_string(tmpid++);
                out << "  " << sb << " = stablehlo.broadcast_in_dim " << s
                    << ", broadcast_dimensions = dense<[0]> : tensor<" << B << "x1xf32> -> "
                    << ty2d(Z->value) << "\n";
                std::string v = newv();
                out << "  " << v << " = stablehlo.divide " << e << ", " << sb
                    << " : " << ty2d(Z->value) << "\n";
                name[n] = v;
                break;
            }
            case Op::LogSumExpRow: {
                Node* Z = n->inputs[0].get();
                std::string zn = name.count(Z) ? name[Z] : (name[Z]=newv());
                int B = Z->value.rows();
                // m = row_max(z)
                std::string ninf = cst_scalar(-std::numeric_limits<float>::infinity());
                std::string m = newv();
                out << "  " << m << " = stablehlo.reduce " << zn << ", " << ninf
                    << " dimensions = [1] : " << ty2d(Z->value) << ", " << ty_scalar()
                    << " -> tensor<" << B << "x1xf32> {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32):\n"
                    << "      %r = stablehlo.maximum %lhs, %rhs : f32\n"
                    << "      stablehlo.return %r : f32\n"
                    << "  }\n";
                // zc = z - m
                std::string mb = "%mb" + std::to_string(tmpid++);
                out << "  " << mb << " = stablehlo.broadcast_in_dim " << m
                    << ", broadcast_dimensions = dense<[0]> : tensor<" << B << "x1xf32> -> "
                    << ty2d(Z->value) << "\n";
                std::string zc = newv();
                out << "  " << zc << " = stablehlo.subtract " << zn << ", " << mb
                    << " : " << ty2d(Z->value) << "\n";
                // e = exp(zc); s = row_sum(e)
                std::string e = newv(); out << "  " << e << " = stablehlo.exponential " << zc << " : " << ty2d(Z->value) << "\n";
                std::string zero = cst_scalar(0.0f);
                std::string s = newv();
                out << "  " << s << " = stablehlo.reduce " << e << ", " << zero
                    << " dimensions = [1] : " << ty2d(Z->value) << ", " << ty_scalar()
                    << " -> tensor<" << B << "x1xf32> {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32):\n"
                    << "      %r = stablehlo.add %lhs, %rhs : f32\n"
                    << "      stablehlo.return %r : f32\n"
                    << "  }\n";
                // lse = log(s) + m
                std::string ls = newv(); out << "  " << ls << " = stablehlo.log " << s << " : tensor<" << B << "x1xf32>\n";
                std::string v = newv(); out << "  " << v << " = stablehlo.add " << ls << ", " << m << " : tensor<" << B << "x1xf32>\n";
                name[n] = v;
                break;
            }

            // ----- Loss -----
            case Op::CeWithLogits: {
                // CE = -mean( sum( Y * (Z - lse(Z)) , axis=1) )
                Node* Z = n->inputs[0].get();
                Node* Y = n->inputs[1].get();
                int B = Z->value.rows();
                std::string zn = name.count(Z) ? name[Z] : (name[Z]=newv());
                std::string yn = name.count(Y) ? name[Y] : (name[Y]=newv());

                // lse
                // (reuse lowering above inline)
                // m = row_max(Z)
                std::string ninf = cst_scalar(-std::numeric_limits<float>::infinity());
                std::string m = newv();
                out << "  " << m << " = stablehlo.reduce " << zn << ", " << ninf
                    << " dimensions = [1] : " << ty2d(Z->value) << ", " << ty_scalar()
                    << " -> tensor<" << B << "x1xf32> {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32):\n"
                    << "      %r = stablehlo.maximum %lhs, %rhs : f32\n"
                    << "      stablehlo.return %r : f32\n"
                    << "  }\n";
                std::string mb = "%mb" + std::to_string(tmpid++);
                out << "  " << mb << " = stablehlo.broadcast_in_dim " << m
                    << ", broadcast_dimensions = dense<[0]> : tensor<" << B << "x1xf32> -> "
                    << ty2d(Z->value) << "\n";
                std::string zc = newv(); out << "  " << zc << " = stablehlo.subtract " << zn << ", " << mb
                                             << " : " << ty2d(Z->value) << "\n";
                std::string e = newv();  out << "  " << e  << " = stablehlo.exponential " << zc << " : " << ty2d(Z->value) << "\n";
                std::string zero = cst_scalar(0.0f);
                std::string s = newv();
                out << "  " << s << " = stablehlo.reduce " << e << ", " << zero
                    << " dimensions = [1] : " << ty2d(Z->value) << ", " << ty_scalar()
                    << " -> tensor<" << B << "x1xf32> {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32):\n"
                    << "      %r = stablehlo.add %lhs, %rhs : f32\n"
                    << "      stablehlo.return %r : f32\n"
                    << "  }\n";
                std::string lse = newv(); out << "  " << lse << " = stablehlo.log " << s << " : tensor<" << B << "x1xf32>\n";
                std::string lsm = newv(); out << "  " << lsm << " = stablehlo.subtract " << zn << ", " << mb
                                              << " : " << ty2d(Z->value) << "\n"; // reuse zc actually
                // prod = Y * log_softmax
                std::string prod = newv(); out << "  " << prod << " = stablehlo.multiply " << yn << ", " << lsm << " : " << ty2d(Z->value) << "\n";
                // rs = row_sum(prod)
                std::string rs = newv();
                out << "  " << rs << " = stablehlo.reduce " << prod << ", " << zero
                    << " dimensions = [1] : " << ty2d(Z->value) << ", " << ty_scalar()
                    << " -> tensor<" << B << "x1xf32> {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32):\n"
                    << "      %r = stablehlo.add %lhs, %rhs : f32\n"
                    << "      stablehlo.return %r : f32\n"
                    << "  }\n";
                // mean over batch and negate
                std::string sB = cst_scalar(-1.0f / float(B));
                std::string ssum = newv(); // sum over batch (dim 0)
                out << "  " << ssum << " = stablehlo.reduce " << rs << ", " << zero
                    << " dimensions = [0] : tensor<" << B << "x1xf32>, " << ty_scalar()
                    << " -> " << ty_scalar() << " {\n"
                    << "    ^bb0(%lhs: f32, %rhs: f32):\n"
                    << "      %r = stablehlo.add %lhs, %rhs : f32\n"
                    << "      stablehlo.return %r : f32\n"
                    << "  }\n";
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
                out << "  " << v << " = stablehlo.copy " << xn << " : " << (X ? ty2d(X->value) : "tensor<?>") << "\n";
                name[n] = v;
                break;
            }
        } // switch
    }

    // Return root
    std::string ret = name[root.node.get()];
    if (ret.empty()) ret = "%UNDEF";
    out << "  return " << ret << " : " << ty2d(root.val()) << "\n";
    out << "  }\n"; // func
    out << "}\n";    // module

    out.close();
    std::cout << "Wrote StableHLO MLIR to: " << filepath << "\n";
}

} // namespace ag::hlo
