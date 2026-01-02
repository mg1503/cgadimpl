// test_mlp.cpp:
#include "ad/ag_all.hpp"
#include "ad/utils/debug.hpp"
#include "ad/utils/export_hlo.hpp"
#include <iostream>

using namespace ag;
using namespace OwnTensor;

int main() {
    std::cout << "\n--- Hand-Calculable MLP Test ---\n\n";

    // --- 1. Set up small, deterministic dimensions ---
    const int B = 2, In = 2, H1 = 2, H2 = 2, H3 = 2, Out = 2;

    // --- 2. Create deterministic data and parameters ---
    // Input X
    Tensor Xt(Shape{{B, In}}, false);
    Xt.data<float>()[0] = 1.0f; Xt.data<float>()[1] = 2.0f;
    Xt.data<float>()[2] = 3.0f; Xt.data<float>()[3] = 4.0f;
    Value X = make_tensor(Xt, "X");
    debug::print_value("Input X", X);

    // Target Y
    Tensor Yt(Shape{{B, Out}}, false);
    Yt.data<float>()[0] = 1.0f; Yt.data<float>()[1] = 0.0f;
    Yt.data<float>()[2] = 0.0f; Yt.data<float>()[3] = 1.0f;
    Value Y = make_tensor(Yt, "Y");
    debug::print_value("Target Y", Y);

    // Layer 1
    Tensor W1t(Shape{{H1, In}}, TensorOptions().with_req_grad(true));
    W1t.data<float>()[0] = 0.1f; W1t.data<float>()[1] = 0.3f;
    W1t.data<float>()[2] = 0.2f; W1t.data<float>()[3] = 0.4f;
    Value W1 = make_tensor(W1t, "W1");
    Tensor b1t(Shape{{1, H1}}, TensorOptions().with_req_grad(true));
    b1t.data<float>()[0] = 0.1f; b1t.data<float>()[1] = 0.1f;
    Value b1 = make_tensor(b1t, "b1");
    debug::print_value("W1", W1);
    debug::print_value("b1", b1);

    // Layer 2
    Tensor W2t(Shape{{H2, H1}}, TensorOptions().with_req_grad(true));
    W2t.data<float>()[0] = 0.5f; W2t.data<float>()[1] = 0.7f;
    W2t.data<float>()[2] = 0.6f; W2t.data<float>()[3] = 0.8f;
    Value W2 = make_tensor(W2t, "W2");
    Tensor b2t(Shape{{1, H2}}, TensorOptions().with_req_grad(true));
    b2t.data<float>()[0] = 0.2f; b2t.data<float>()[1] = 0.2f;
    Value b2 = make_tensor(b2t, "b2");
    debug::print_value("W2", W2);
    debug::print_value("b2", b2);

    // Layer 3
    Tensor W3t(Shape{{H3, H2}}, TensorOptions().with_req_grad(true));
    W3t.data<float>()[0] = 0.1f; W3t.data<float>()[1] = 0.2f;
    W3t.data<float>()[2] = 0.3f; W3t.data<float>()[3] = 0.4f;
    Value W3 = make_tensor(W3t, "W3");
    Tensor b3t(Shape{{1, H3}}, TensorOptions().with_req_grad(true));
    b3t.data<float>()[0] = 0.3f; b3t.data<float>()[1] = 0.3f;
    Value b3 = make_tensor(b3t, "b3");
    debug::print_value("W3", W3);
    debug::print_value("b3", b3);

    // Layer 4 (Output)
    Tensor W4t(Shape{{Out, H3}}, TensorOptions().with_req_grad(true));
    W4t.data<float>()[0] = 0.5f; W4t.data<float>()[1] = 0.6f;
    W4t.data<float>()[2] = 0.7f; W4t.data<float>()[3] = 0.8f;
    Value W4 = make_tensor(W4t, "W4");
    Tensor b4t(Shape{{1, Out}}, TensorOptions().with_req_grad(true));
    b4t.data<float>()[0] = 0.4f; b4t.data<float>()[1] = 0.4f;
    Value b4 = make_tensor(b4t, "b4");
    debug::print_value("W4", W4);
    debug::print_value("b4", b4);

    // --- 3. Forward Pass with Simplified Activations and Verbose Printing ---
    std::cout << "\n--- FORWARD PASS ---\n";
    Value L1 = ag::relu(ag::linear(X,  W1, b1));
    debug::print_value("L1 = relu(linear(X, W1, b1))", L1);

    Value L2 = ag::relu(ag::linear(L1, W2, b2));
    debug::print_value("L2 = relu(linear(L1, W2, b2))", L2);

    Value L3 = ag::relu(ag::linear(L2, W3, b3));
    debug::print_value("L3 = relu(linear(L2, W3, b3))", L3);
    
    Value logits = ag::linear(L3, W4, b4);
    debug::print_value("logits = linear(L3, W4, b4)", logits);


    // --- 4. Loss and Backward Pass ---
    Value loss = ag::mse_loss(logits, Y);

    ag::debug::dump_dot(loss, "graph_mlp.jpg");
    backward(loss);
    ag::hlo::dump_stablehlo(loss,  "build/mlp_test_hlo.mlir");
    // --- 5. Print All Gradients ---
    std::cout << "\n--- RESULTS ---\n";
    debug::print_value("Final Loss", loss);
    debug::print_grad("W1", W1);
    debug::print_grad("b1", b1);
    debug::print_grad("W2", W2);
    debug::print_grad("b2", b2);
    debug::print_grad("W3", W3);
    debug::print_grad("b3", b3);
    debug::print_grad("W4", W4);
    debug::print_grad("b4", b4);
    
    std::cout << "\n  test_mlp completed.\n";

    /*
    --- HAND CALCULATION FOR L1 ---
    ag::linear(X, W1, b1) computes (X @ W1.T) + b1

    X [2, 2] = [[1, 2],
                [3, 4]]
    
    W1 [2, 2] = [[0.1, 0.3],
                 [0.2, 0.4]]
    
    W1.T [2, 2] = [[0.1, 0.2],
                   [0.3, 0.4]]

    X @ W1.T [2, 2] = [[1*0.1+2*0.3, 1*0.2+2*0.4],
                       [3*0.1+4*0.3, 3*0.2+4*0.4]]
                    = [[0.7, 1.0],
                       [1.5, 2.2]]

    (X @ W1.T) + b1 = [[0.7, 1.0],   + [[0.1, 0.1],
                       [1.5, 2.2]]      [0.1, 0.1]] (broadcasted)
                    = [[0.8, 1.1],
                       [1.6, 2.3]]
    
    L1 = relu(...) = [[0.8, 1.1],
                      [1.6, 2.3]]  (All positive, so no change)
    */

    return 0;
}