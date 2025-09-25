// tests/test_complex_mlp.cpp
#include <iostream>
#include <random>
#include "ad/ag_all.hpp"
#include <iomanip>


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


using namespace ag;

int main() {
    // ag::debug::enable_tracing(true);
    // ag::debug::set_print_limits(/*rows*/6, /*cols*/8, /*width*/10, /*precision*/4);

    // ---------- Shapes ----------
    const int B = 8;     // batch size
    const int In = 16;   // input dim
    const int H1 = 64;
    const int H2 = 64;
    const int H3 = 32;
    const int H4 = 32;
    const int Out = 10;  // number of classes

    // ---------- Data ----------
    Tensor Xt = Tensor::randn(B, In, /*seed=*/123);
    Value  X  = constant(Xt, "X");  // inputs are constants (no grads)

    // One-hot labels Y[B,Out]
    Tensor Yt(B, Out);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> pick(0, Out - 1);
    for (int i = 0; i < B; ++i) {
        int k = pick(gen);
        for (int j = 0; j < Out; ++j) Yt(i, j) = (j == k) ? 1.f : 0.f;
    }
    Value Y = constant(Yt, "Y");

    // ---------- Parameters ----------
    // W[k][in,out], b[k][1,out]
    auto W1 = param(Tensor::randn(In,  H1, 1001), "W1");
    auto b1 = param(Tensor::zeros(1,   H1),       "b1");
    // print_value("X", X);
    // print_value("W1", W1);
    // print_value("b1", b1);

    auto W2 = param(Tensor::randn(H1,  H2, 1002), "W2");
    auto b2 = param(Tensor::zeros(1,   H2),       "b2");

    auto W3 = param(Tensor::randn(H2,  H3, 1003), "W3");
    auto b3 = param(Tensor::zeros(1,   H3),       "b3");

    auto W4 = param(Tensor::randn(H3,  H4, 1004), "W4");
    auto b4 = param(Tensor::zeros(1,   H4),       "b4");

    auto W5 = param(Tensor::randn(H4,  Out, 1005), "W5");
    auto b5 = param(Tensor::zeros(1,   Out),       "b5");

    // ---------- Forward: 4 hidden layers + logits ----------
    // L1 = GELU(X @ W1 + b1)
    Value L1 = gelu( matmul(X,  W1) + b1 );          // [B,H1]
    // print_value("L1", L1);

    // L2 = SiLU(L1 @ W2 + b2)
    Value L2 = silu( matmul(L1, W2) + b2 );          // [B,H2]
    // print_value("L2", L2);

    // L3 = LeakyReLU(L2 @ W3 + b3, alpha=0.1)
    Value L3_pre = fmab(L2, W3, b3);              // [B,H3]
    Value L3 = leaky_relu(L3_pre, 0.1f);             // [B,H3]
    // print_value("L3", L3); 

    // L4 = Softplus(L3 @ W4 + b4)
    Value L4 = softplus( fmab(L3, W4,  b4) );      // [B,H4]
    // print_value("L4", L4);

    // Logits = L4 @ W5 + b5
    Value logits = matmul(L4, W5) + b5;              // [B,Out]
    // print_value("logits", logits);

    // Loss: stable cross-entropy with logits (one-hot)
    Value loss = cross_entropy_with_logits(logits, Y); // scalar [1,1]
    // print_value("Y (one-hot)", Y);
    // print_value("loss", loss);

    // ---------- Backprop ----------
    zero_grad(loss);
    backward(loss);
    
    // ag::debug::print_all_grads(loss);                   // optional full gradient dump
    // ag::debug::dump_dot(loss, "build/graph.dot");       // write GraphViz DOT
    // ag::debug::dump_vjp_dot(loss, "build/graph_vjp.dot"); // write VJP DOT
    // ag::debug::dump_jvp_dot(loss, "build/graph_jvp.dot"); // write JVP DOT



    // ---------- Report ----------
    std::cout << "loss = " << loss.val()(0,0) << "\n";

    // Show a few logits + softmax probs for the first row
    Value probs = softmax_row(logits);
    std::cout << "logits[0,:0..4] = ";
    for (int j = 0; j < std::min(5, Out); ++j)
        std::cout << logits.val()(0, j) << (j+1<5? ' ' : '\n');

    std::cout << "probs [0,:0..4] = ";
    for (int j = 0; j <Out; ++j)
        std::cout << probs.val()(0, j) << (j+1<10? ' ' : '\n');

    // Peek at some gradient entries so you know it's flowing
    std::cout << "dL/dW1[0,0]=" << W1.grad()(0,0)
              << "  dL/db1[0,0]=" << b1.grad()(0,0) << "\n";
    // print_grad("W1", W1);
    // print_grad("b1", b1);
    std::cout << "dL/dW2[0,0]=" << W2.grad()(0,0)
              << "  dL/db2[0,0]=" << b2.grad()(0,0) << "\n";
    // print_grad("W2", W2);
    // print_grad("b2", b2);
    std::cout << "dL/dW3[0,0]=" << W3.grad()(0,0)
              << "  dL/db3[0,0]=" << b3.grad()(0,0) << "\n";
    // print_grad("W3", W3);
    // print_grad("b3", b3);
    std::cout << "dL/dW4[0,0]=" << W4.grad()(0,0)
              << "  dL/db4[0,0]=" << b4.grad()(0,0) << "\n";
    // print_grad("W4", W4);   
    // print_grad("b4", b4);
    std::cout << "dL/dW5[0,0]=" << W5.grad()(0,0)
              << "  dL/db5[0,0]=" << b5.grad()(0,0) << "\n";
    // print_grad("W5", W5);   
    // print_grad("b5", b5);

    // ---------- (Optional) one SGD step, in-place (no extra graph) ----------
    // NOTE: we mutate raw tensor values via operator()(i,j) to avoid building a graph.
    // float lr = 0.05f;
    // {
    //     auto upd = [lr](Value& P){
    //         auto [R,C] = P.shape();
    //         for (int i=0;i<R;++i) for (int j=0;j<C;++j)
    //             P.node->value(i,j) -= lr * P.node->grad(i,j);
    //     };
    //     upd(W1); upd(b1);
    //     upd(W2); upd(b2);
    //     upd(W3); upd(b3);
    //     upd(W4); upd(b4);
    //     upd(W5); upd(b5);
    // }

    // std::cout << "SGD step applied (lr=" << lr << ")\n";
    return 0;
}
