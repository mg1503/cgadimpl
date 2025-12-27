#include <iostream>
#include "ad/ag_all.hpp"
#include "optim.hpp"
#include <random>
#include <iomanip>
using namespace ag;


int main() {
    using namespace std;

    // --- 1. Create Tensors (This part is correct) ---
    Tensor A_tensor = Tensor::randn(Shape{{2, 3}}, TensorOptions().with_req_grad(true));
    Tensor B_tensor = Tensor::randn(Shape{{3, 2}}, TensorOptions().with_req_grad(true));
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

        auto q = ag::matmul(a, b) + bias;
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
        
        std::cout << "--- TENSOR PRINT UTILITY USAGE ---" << std::endl;
        std::cout << "Printing q value:" << std::endl;
        // q is a Value, so we access .val() which returns a Tensor&
        q.val().display();

        std::cout << "Printing q gradient:" << std::endl;
        // q.grad() returns a Tensor (view of gradient), so we can call .display() on it directly
        q.grad().display();
        cout << "\n--- Backward Pass Results ---\n";
        std::cout<< "Grad of y (dL/dy) "<<std::endl;
        y.val().display();

        debug::print_grad("\nGrad of b (dL/dB)", b); // Correct
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