// // =====================
// // file: tests/test_ag.cpp
// // =====================
// #include <iostream>
// #include "ad/ag_all.hpp"
// #include <iomanip>
// #include <iostream>

// static void printTensor(const char* name,
//                         const ag::Tensor& T,
//                         int max_r = -1, int max_c = -1,
//                         int width = 9, int prec = 4) {
//     using std::cout;
//     using std::fixed;
//     using std::setw;
//     using std::setprecision;

//     const int r = T.rows(), c = T.cols();
//     if (max_r < 0) max_r = r;
//     if (max_c < 0) max_c = c;

//     cout << name << " [" << r << "x" << c << "]";
//     if (r == 1 && c == 1) { // scalar fast path
//         cout << " = " << fixed << setprecision(6) << T(0,0) << "\n";
//         return;
//     }
//     cout << "\n";

//     const int rr = std::min(r, max_r);
//     const int cc = std::min(c, max_c);
//     for (int i = 0; i < rr; ++i) {
//         cout << "  ";
//         for (int j = 0; j < cc; ++j) {
//             cout << setw(width) << fixed << setprecision(prec) << T(i,j);
//         }
//         if (cc < c) cout << " ...";
//         cout << "\n";
//     }
//     if (rr < r) cout << "  ...\n";
// }

// using namespace std;

// int main(){
// using namespace ag;
// Tensor A = Tensor::randn(2,3);
// Tensor B = Tensor::randn(3,2);
// auto a = param(A, "A");
// auto b = param(B, "B");


// auto y = sum(relu(matmul(a,b))); // scalar


// zero_grad(y);
// backward(y);
// std::cout << "y = " << y.val().sum_scalar() << endl;
// std::cout << "dL/dA[0,0] = " << a.grad()(0,0) << ", dL/dB[0,0] = " << b.grad()(0,0) << endl;


// // JVP: along dA=ones, dB=zeros
// std::unordered_map<Node*, Tensor> seed; seed[a.node.get()] = Tensor::ones_like(a.val());
// Tensor jy = jvp(y, seed);
// std::cout << "JVP dy(dA,0) = " << jy(0,0) << endl;

// printTensor("A", a.val());
// printTensor("B", b.val());
// ag::Tensor Z = ag::Tensor::matmul(a.val(), b.val());
// printTensor("Z = A*B", Z);
// printTensor("ReLU mask", ag::Tensor::relu_mask(Z));
// printTensor("grad A", a.grad());
// printTensor("grad B", b.grad());
// printTensor("JVP dy(dA,0)", jy);  // jy is 1x1, prints as scalar

// cout << "Numerically verified! \nTest successful!\n";
// return 0;
// }
#include <iostream>
#include "ad/ag_all.hpp"
#include "optim.hpp"
#include <random>
#include <iomanip>
using namespace ag;


int main() {
    using namespace std;
    // using namespace ag;

    // --- 1. Correct Tensor and Value Creation ---

    // Create a Tensor that requires a gradient for a model input 'a'
    Tensor A_tensor = Tensor::randn(Shape{{2, 3}}, TensorOptions().with_req_grad(true));
    auto a = make_tensor(A_tensor, "A");

    // Create Tensors for trainable parameters 'b' and 'bias'
    Tensor B_tensor = Tensor::randn(Shape{{3, 2}}, TensorOptions().with_req_grad(true));
    auto b = make_tensor(B_tensor, "B");
    
    Tensor Bias_tensor = Tensor::zeros(Shape{{1, 2}}, TensorOptions().with_req_grad(true));
    auto bias = make_tensor(Bias_tensor, "bias");

    // Create the target tensor 'Yt'. This is a constant, so requires_grad=false (the default).
    Tensor Yt(Shape{{2, 2}}, TensorOptions());
    float* yt_data = Yt.data<float>(); // Get the raw data pointer
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> pick(0, 1); // Corrected range for 2 columns
    for (int i = 0; i < 2; ++i) {
        int k = pick(gen);
        for (int j = 0; j < 2; ++j) {
            // Access elements using linear indexing: row * num_cols + col
            yt_data[i * 2 + j] = (j == k) ? 1.0f : 0.0f;
        }
    }
    // Wrap the constant tensor in a Value node
    auto W = make_tensor(Yt, "Y_target");

    // --- 2. Training Loop ---

    cout << fixed << setprecision(4);
    for (int i = 0; i < 10; ++i) {
        cout << "\n=============== Iteration " << i << " ===============\n";

        // Forward pass
        auto q = matmul(a, b) + bias;
        // Note: The original code added bias twice. I have preserved this logic.
        auto y = kldivergence(q + bias, W);

        // --- Before backward ---
        cout << "--- Forward Pass Results ---\n";
        debug::print_value("Loss (y)", y);
        debug::print_grad("Grad of b (before)", b);
        debug::print_grad("Grad of bias (before)", bias);

        // Zero gradients from previous iteration
        zero_grad(y);

        // Backward pass to compute gradients
        backward(y);

        // --- After backward ---
        cout << "\n--- Backward Pass Results ---\n";
        debug::print_grad("Grad of y (dL/dy)", y); // Will be 1.0
        debug::print_grad("Grad of b (dL/dB)", b);
        debug::print_grad("Grad of bias (dL/dbias)", bias);

        // Update weights using the optimizer
        SGD(y, nullptr, 0.1f); // Using a learning rate of 0.1

        // --- After SGD update ---
        cout << "\n--- After SGD Step ---\n";
        debug::print_value("Updated b", b);
        debug::print_value("Updated bias", bias);
    }

    return 0;
}