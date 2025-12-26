#include "ad/ag_all.hpp"
#include <iostream>
#include <cmath>

using namespace ag;
using namespace OwnTensor;

// Helper to print a value using the modern display method
void print_val(const std::string& label, const Value& v) {
    std::cout << label << ":\n";
    v.val().display(std::cout, 4);
    std::cout << "\n";
}

void print_grad(const std::string& label, const Value& v) {
    std::cout << label << ".grad:\n";
    v.grad().display(std::cout, 4);
    std::cout << "\n";
}

int main() {
    std::cout << "=== Tiny hand-checkable MLP (2x2) ===\n";
    const int B = 2, In = 2, H = 2, Out = 2;

    Tensor Xt(Shape{{B, In}}, TensorOptions());
    float* x_data = Xt.data<float>();
    x_data[0*In + 0] = 1; x_data[0*In + 1] = -1;
    x_data[1*In + 0] = 0; x_data[1*In + 1] = 2;
    Value X = make_tensor(Xt, "X");
    print_val("X", X);

    Tensor Yt(Shape{{B, Out}}, false);
    float* y_data = Yt.data<float>();
    y_data[0*Out + 0] = 1; y_data[0*Out + 1] = 0;
    y_data[1*Out + 0] = 0; y_data[1*Out + 1] = 1;
    Value Y = make_tensor(Yt, "Y");
    print_val("Y (one-hot)", Y);

    Tensor W1t(Shape{{In, H}}, TensorOptions().with_req_grad(true));
    float* w1_data = W1t.data<float>();
    w1_data[0*H + 0] = 2; w1_data[0*H + 1] = 0;
    w1_data[1*H + 0] = -1; w1_data[1*H + 1] = 1;
    Value W1 = make_tensor(W1t, "W1");

    Tensor b1t(Shape{{1, H}}, TensorOptions().with_req_grad(true));
    b1t.data<float>()[0] = 1; b1t.data<float>()[1] = -2;
    Value b1 = make_tensor(b1t, "b1");

    Value L1   = relu(matmul(X, W1) + b1);

    Tensor W2t(Shape{{H, Out}}, TensorOptions().with_req_grad(true));
    float* w2_data = W2t.data<float>();
    w2_data[0*Out + 0] = 1; w2_data[0*Out + 1] = -1;
    w2_data[1*Out + 0] = 2; w2_data[1*Out + 1] = 0;
    Value W2 = make_tensor(W2t, "W2");

    Tensor b2t(Shape{{1, Out}}, TensorOptions().with_req_grad(true));
    b2t.data<float>()[0] = 0; b2t.data<float>()[1] = 1;
    Value b2 = make_tensor(b2t, "b2");

    Value logits = matmul(L1, W2) + b2;
    Value loss   = cross_entropy_with_logits(logits, Y);
    print_val("loss", loss);

    backward(loss);
    print_grad("W1", W1);
    print_grad("W2", W2);

    std::cout << "=== Done: tiny hand-checkable test ===\n";
    return 0;
}