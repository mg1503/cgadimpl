#include <iostream>
#include <random>
#include "ad/ag_all.hpp"
#include "ad/export_hlo.hpp"

#include <iomanip>
#include <fstream>
#include <filesystem>
// ==========================================================================
// CSV dump utilities
// ==========================================================================
// 
// static void write_csv_tensor(const ag::Tensor& T,
//                              const std::string& filepath,
//                              int precision = 6)
// {
//     std::filesystem::create_directories(std::filesystem::path(filepath).parent_path());
//     std::ofstream out(filepath);
//     if (!out) { std::cerr << "Failed to open " << filepath << "\n"; return; }
//     out << std::fixed << std::setprecision(precision);
//     for (int i = 0; i < T.rows(); ++i) {
//         for (int j = 0; j < T.cols(); ++j) {
//             if (j) out << ',';
//             out << T(i, j);
//         }
//         out << '\n';
//     }
//     out.close();
//     std::cout << "Wrote " << T.rows() << "x" << T.cols()
//               << " CSV to: " << filepath << "\n";
// }

// // Convenience wrappers for ag::Value
// static void dump_csv_val (const std::string& label, const ag::Value& v,
//                           const std::string& dir = "build/dumps")
// {
//     write_csv_tensor(v.val(),  dir + "/" + label + ".csv");
// }
// static void dump_csv_grad(const std::string& label,  ag::Value& v,
//                           const std::string& dir = "build/dumps")
// {
//     write_csv_tensor(v.grad(), dir + "/" + label + "_grad.csv");
// }


// ==========================================================================
// Pretty-print utilities
// ==========================================================================

static void print_tensor(const std::string& label,
                         const ag::Tensor& T,
                         int max_r = 6, int max_c = 8,
                         int width = 10, int precision = 4) {
    std::cout << label << " [" << T.rows() << "x" << T.cols() << "]\n";
    const int R = std::min(T.rows(), max_r);
    const int C = std::min(T.cols(), max_c);
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            std::cout << std::setw(width) << std::fixed
                      << std::setprecision(precision) << T(i,j);
        }
        if (C < T.cols()) std::cout << "  ...";
        std::cout << "\n";
    }
    if (R < T.rows()) std::cout << "...\n";
    std::cout << std::endl;
}

static void print_value(const std::string& label,
                        const ag::Value& v,
                        int max_r = 6, int max_c = 8) {
    print_tensor(label, v.val(), max_r, max_c);
}
static void print_grad(const std::string& label, ag::Value& v,  int max_r = 6, int max_c = 8) {
    print_tensor(label + ".grad", v.grad(), max_r, max_c);
}

// ==========================================================================
// MLP utilities
// ==========================================================================

using namespace ag;

// Linear layer: X[B,in] @ W[in,out] + b[1,out]
static inline ag::Value linear(const ag::Value& X, ag::Value& W, ag::Value& b) {
    return matmul(X, W) + b; // b broadcasts row-wise
}

// Mean-squared error over all elements (batch & classes)
static inline ag::Value mse_lossa(const ag::Value& pred, const ag::Value& target) {
    ag::Value diff = pred - target;
    ag::Value sq   = diff * diff;               // elementwise
    ag::Value s    = sum(sq);                   // scalar [1,1]
    int B = pred.shape().first, C = pred.shape().second;
    ag::Tensor scale = ag::Tensor::ones(1,1);
    scale(0,0) = 1.0f / float(B * C);
    return s * constant(scale);                 // broadcast scalar
}


int main() {
    // ----- Shapes (tweak freely) -----
    const int B = 4;     // batch
    const int In = 8;    // input dim
    const int H1 = 32;
    const int H2 = 32;
    const int H3 = 16;
    const int Out = 10;  // classes

    // ----- Data -----
    ag::Tensor Xt = ag::Tensor::randn(B, In);
    ag::Value  X  = constant(Xt, "X");          // treat input as constant (no grads)

    // One-hot labels Y[B,Out]
    ag::Tensor Yt(B, Out);
    std::mt19937 gen(123);
    std::uniform_int_distribution<int> cls(0, Out-1);
    for (int i=0;i<B;++i) {
        int k = cls(gen);
        for (int j=0;j<Out;++j) Yt(i,j) = (j==k) ? 1.f : 0.f;
    }
    ag::Value Y = constant(Yt, "Y");
    

    // ----- Parameters -----
    // Weights W*[in,out], biases b*[1,out]
    auto W1 = param(ag::Tensor::randn(In, H1),  "W1");
    auto b1 = param(ag::Tensor::zeros(1, H1),   "b1");
    print_value("X", X);
    print_value("W1", W1);
    print_value("b1", b1);

    auto W2 = param(ag::Tensor::randn(H1, H2),  "W2");
    auto b2 = param(ag::Tensor::zeros(1, H2),   "b2");

    auto W3 = param(ag::Tensor::randn(H2, H3),  "W3");
    auto b3 = param(ag::Tensor::zeros(1, H3),   "b3");

    auto W4 = param(ag::Tensor::randn(H3, Out), "W4");
    auto b4 = param(ag::Tensor::zeros(1, Out),  "b4");

    // ----- Forward (4 layers): X -> L1 -> L2 -> L3 -> logits -----
    ag::Value L1 = relu(fmab(X,  W1, b1));              // [B,H1]
    print_value("L1", L1);
    ag::Value L2 = relu(fmab(L1, W2, b2));              // [B,H2]
    print_value("L2", L2);
    ag::Value L3 = gcu(fmab(L2, W3, b3));              // [B,H3]
    print_value("L3", L3);  
    ag::Value logits =       fmab(L3, W4, b4);          // [B,Out]
    print_value("logits", logits);

    // NOTE: Using MSE on logits for now (no softmax yet).
    ag::Value loss = mse_loss(logits, Y);                 // scalar [1,1]
    print_value("Y(one-hot)", Y);
    
    
    // // After forward:
    // dump_csv_val("X", X);
    // dump_csv_val("L1", L1);
    // dump_csv_val("L2", L2);
    // dump_csv_val("L3", L3);
    // dump_csv_val("logits", logits);
    // dump_csv_val("Y", Y);

    ag::hlo::dump_stablehlo(loss, "build/graph_stablehlo.mlir");

    // ----- Backprop -----
    zero_grad(loss);
    backward(loss);
// ag::hlo::dump_stablehlo(loss, "build/graph_stablehlo.mlir");

    // ----- Prints -----
    std::cout << "loss = " << loss.val()(0,0) << "\n";
    std::cout << "dL/dW1[0,0] = " << W1.grad()(0,0)
              << ", dL/db1[0,0] = " << b1.grad()(0,0) << "\n";
    print_grad("W1", W1);
    print_grad("b1", b1);
    
    std::cout << "dL/dW2[0,0] = " << W2.grad()(0,0)
              << ", dL/db2[0,0] = " << b2.grad()(0,0) << "\n";
    print_grad("W2", W2);
    print_grad("b2", b2);

    std::cout << "dL/dW3[0,0] = " << W3.grad()(0,0)
              << ", dL/db3[0,0] = " << b3.grad()(0,0) << "\n";
    print_grad("W3", W3);
    print_grad("b3", b3);
    
    std::cout << "dL/dW4[0,0] = " << W4.grad()(0,0)
              << ", dL/db4[0,0] = " << b4.grad()(0,0) << "\n";
    print_grad("W4", W4);
    print_grad("b4", b4);

        
    // // After backprop:
    // dump_csv_grad("W1", W1);
    // dump_csv_grad("b1", b1);
    // dump_csv_grad("W2", W2);
    // dump_csv_grad("b2", b2);
    // dump_csv_grad("W3", W3);
    // dump_csv_grad("b3", b3);
    // dump_csv_grad("W4", W4);
    // dump_csv_grad("b4", b4);

    // Optional: one SGD step (no graph build if you do it as raw tensor math)
    // float lr = 0.01f;
    // for(auto* P : {W1.node.get(), W2.node.get(), W3.node.get(), W4.node.get(),
    //                b1.node.get(), b2.node.get(), b3.node.get(), b4.node.get()}) {
    //     for (size_t i=0;i<P->value.size();++i) P->value.d[i] -= lr * P->grad.d[i];
    // }

    return 0;
}
