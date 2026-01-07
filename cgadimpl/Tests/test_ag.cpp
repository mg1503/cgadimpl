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

    // --- 1. Create Tensors (This part is correct) ---
    Tensor A_tensor = Tensor::randn<float>(Shape{{2, 3}}, TensorOptions().with_req_grad(true));
    Tensor B_tensor = Tensor::randn<float>(Shape{{3, 2}}, TensorOptions().with_req_grad(true));
    Tensor Bias_tensor = Tensor::zeros(Shape{{1, 2}}, TensorOptions().with_req_grad(true));
    Tensor Yt_tensor = Tensor::zeros(Shape{{2, 2}}, TensorOptions()); // Target tensor
    // ... (logic to fill Yt_tensor is fine) ...
    // --- OMITTED FOR BREVITY ---
    
    // --- 2. Create Graph HANDLES (Value objects) ---
    // These are the objects we will use to interact with the graph
    auto a = make_tensor(A_tensor, "A");
    auto b = make_tensor(B_tensor, "B");
    auto bias = make_tensor(Bias_tensor, "bias");
    auto W = make_tensor(Yt_tensor, "Y_target");

    // --- 3. Training Loop ---
    cout << fixed << setprecision(4);
    for (int i = 0; i < 10; ++i) {
        cout << "\n=============== Iteration " << i << " ===============\n";

        auto q = matmul(a, b) + bias;
        auto y = kldivergence(q + bias, W);

        cout << "--- Forward Pass Results ---\n";
        debug::print_value("Loss (y)", y);

        // --- FIX: Inspect the grad via the VALUE handle, not the TENSOR handle ---
        debug::print_grad("Grad of b (before)", b);
        debug::print_grad("Grad of bias (before)", bias);
        ag::debug::dump_dot(y, "graph.jpg");

        zero_grad(y);
        backward(y);
        ag::debug::dump_vjp_dot(y, "vjp.jpg");

        cout << "\n--- Backward Pass Results ---\n";
        debug::print_grad("Grad of y (dL/dy)", y);
        debug::print_grad("Grad of b (dL/dB)", b); // Correct
        debug::print_grad("Grad of bias (dL/dbias)", bias); // Correct

        SGD(y, nullptr, 0.1f);

        cout << "\n--- After SGD Step ---\n";
        // The SGD step modifies the .value tensor INSIDE the node.
        // We inspect it through the Value handle.
        debug::print_value("Updated b", b);
        debug::print_value("Updated bias", bias);

        // --- OPTIONAL: If you want the original tensors to be updated ---
        // You can manually copy the data back after the SGD step.
        // This is not necessary for the framework to function, only for inspection.
        B_tensor.copy_(b.val());
        Bias_tensor.copy_(bias.val());
    }

    return 0;
}