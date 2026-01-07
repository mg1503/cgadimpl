#include <iostream>
#include <random>
#include <iomanip>
#include "ad/runtime/jit_compiler.hpp"
#include "ad/ag_all.hpp" // Main umbrella header

using namespace ag;

int main() {
    // For pretty printing
    std::cout << std::fixed << std::setprecision(4);

    // ---------- Shapes ----------
    const int B = 8;     // batch size
    const int In = 16;   // input dim
    const int H1 = 64;
    const int H2 = 64;
    const int H3 = 32;
    const int H4 = 32;
    const int Out = 10;  // number of classes

    // ---------- Data ----------
    // Inputs are constants (requires_grad=false, which is the default)
    Tensor Xt = Tensor::randn<float>(Shape{{B, In}}, TensorOptions());
    Value  X  = make_tensor(Tensor::randn<float>(Shape{{B, In}}, TensorOptions()), "X");

    // One-hot labels Y[B,Out]
    Tensor Yt(Shape{{B, Out}}, TensorOptions());
    float* yt_data = Yt.data<float>(); // Get data pointer to fill
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> pick(0, Out - 1);
    for (int i = 0; i < B; ++i) {
        int k = pick(gen);
        for (int j = 0; j < Out; ++j) {
            yt_data[i * Out + j] = (j == k) ? 1.0f : 0.0f; // Use linear indexing
        }
    }
    Value Y = make_tensor(Yt, "Y_target");

    // ---------- Parameters ----------
    // Parameters are trainable (requires_grad=true)
    auto opts_param = TensorOptions().with_req_grad(true);
    auto W1 = make_tensor(Tensor::randn<float>(Shape{{In, H1}}, opts_param), "W1");
    auto b1 = make_tensor(Tensor::zeros(Shape{{1, H1}}, opts_param), "b1");
    auto W2 = make_tensor(Tensor::randn<float>(Shape{{H1, H2}}, opts_param), "W2");
    auto b2 = make_tensor(Tensor::zeros(Shape{{1, H2}}, opts_param), "b2");
    auto W3 = make_tensor(Tensor::randn<float>(Shape{{H2, H3}}, opts_param), "W3");
    auto b3 = make_tensor(Tensor::zeros(Shape{{1, H3}}, opts_param), "b3");
    auto W4 = make_tensor(Tensor::randn<float>(Shape{{H3, H4}}, opts_param), "W4");
    auto b4 = make_tensor(Tensor::zeros(Shape{{1, H4}}, opts_param), "b4");
    auto W5 = make_tensor(Tensor::randn<float>(Shape{{H4, Out}}, opts_param), "W5");
    auto b5 = make_tensor(Tensor::zeros(Shape{{1, Out}}, opts_param), "b5");
    
    // ---------- Forward: 4 hidden layers + logits ----------
    Value L1 = gelu(matmul(X,  W1) + b1);
    Value L2 = silu(matmul(L1, W2) + b2);
    Value L3 = leaky_relu(matmul(L2, W3) + b3, 0.1f);
    Value L4 = softplus(matmul(L3, W4) + b4);
    Value logits = matmul(L4, W5) + b5;
    Value loss = cross_entropy_with_logits(logits, Y);

    // ---------- Backprop ----------
    zero_grad(loss);
    backward(loss);
        // Tell the compiler which leaves are runtime inputs vs. trainable parameters
    std::vector<Value> inputs = {X};
    std::vector<Value> params = {W1, b1, W2,b2,W3,b3,W4,b4,W5,b5};

    // The 'loss' Value is the root of the graph to be compiled
    auto comp = ag::jit::compile(loss, inputs, params);
    
    // Use the framework's debug utilities
    // ag::debug::print_all_grads(loss);
    // ag::debug::dump_dot(loss, "build/graph.dot");
    // ag::debug::dump_vjp_dot(loss, "build/graph_vjp.dot");

    // ---------- Report ----------
    // To get a scalar value, move to CPU and get the data pointer
    float loss_val = loss.val().to_cpu().data<float>()[0];
    std::cout << "loss = " << loss_val << "\n";

    // Show a few logits + softmax probs for the first row
    Value probs = softmax_row(logits);
    Tensor probs_cpu = probs.val().to_cpu();
    Tensor logits_cpu = logits.val().to_cpu();
    const float* probs_data = probs_cpu.data<float>();
    const float* logits_data = logits_cpu.data<float>();

    std::cout << "logits[0,:5] = ";
    for (int j = 0; j < std::min(5, Out); ++j)
        std::cout << logits_data[j] << (j + 1 < 5 ? ' ' : '\n');

    std::cout << "probs [0,:5] = ";
    for (int j = 0; j < std::min(5, Out); ++j)
        std::cout << probs_data[j] << (j + 1 < 5 ? ' ' : '\n');

    // Print gradients using the framework's tools
    std::cout << "\n--- Sample Gradients ---\n";
    debug::print_grad("W1", W1);
    debug::print_grad("b1", b1);
    debug::print_grad("W5", W5);
    debug::print_grad("b5", b5);

    return 0;
}