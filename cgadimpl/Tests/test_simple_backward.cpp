#include <iostream>
#include "ad/ag_all.hpp"
using namespace ag;

int main() {
    std::cout << "Creating tensors..." << std::endl;
    Tensor A_tensor = Tensor::randn(Shape{{2, 2}}, TensorOptions().with_req_grad(true));
    Tensor B_tensor = Tensor::randn(Shape{{2, 2}}, TensorOptions().with_req_grad(true));
    
    auto a = make_tensor(A_tensor, "A");
    auto b = make_tensor(B_tensor, "B");
    
    std::cout << "Forward pass..." << std::endl;
    auto c = a * b;
    auto loss = sum(c);
    
    std::cout << "Loss value: " << loss.val() << std::endl;
    std::cout << "Starting zero_grad..." << std::endl;
    zero_grad(loss);
    
    std::cout << "Starting backward..." << std::endl;
    backward(loss);
    
    std::cout << "Backward complete!" << std::endl;
    std::cout << "Grad A: " << a.grad() << std::endl;
    std::cout << "Grad B: " << b.grad() << std::endl;
    
    return 0;
}
