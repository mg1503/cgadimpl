#include <iostream>
#include "ad/ag_all.hpp"
#include "optim.hpp"
#include <random>
#include <iomanip>
using namespace ag;

using namespace std;

int main(){
    Tensor A_tensor = Tensor::randn(Shape{{2, 2}}, TensorOptions().with_req_grad(true));
    Tensor B_tensor = Tensor::randn(Shape{{2, 2}}, TensorOptions().with_req_grad(true));
    
    Tensor bias_tensor = Tensor::randn(Shape{{1, 2}}, TensorOptions().with_req_grad(true));
    Tensor Yt_tensor = Tensor::zeros(Shape{{2, 2}}, TensorOptions());

    auto a = make_tensor(A_tensor, "X");
    auto b = make_tensor(B_tensor, "W");

    auto bias = make_tensor(bias_tensor, "bi");
    auto tar = make_tensor(Yt_tensor, "y");

    Value L1 = linear(a, b, bias);
    Value a1 = softmax_row(L1);
    Value loss = mae_loss(a1, tar);

    backward(loss);

    debug::print_grad("loss", a);
    debug::print_grad("loss", b);
    debug::print_value("Loss (y)", loss);
    debug::print_value("a1", a1);
    debug::print_value("L1", L1);
    return 0;
}