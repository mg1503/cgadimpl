#include <ad/ag_all.hpp>
#include "optim.hpp"
#include <iostream>

int main(){
    std::cout << "Building computational graph with 3 branches, 10 layers each..." << std::endl;
    // Weights and biases for each branch (all output size 2048)
    auto w1 = ag::make_tensor(OwnTensor::Tensor::randn<float>(Shape{{2048,2048}}, OwnTensor::TensorOptions().with_req_grad(true)), "w1");
    auto b1 = ag::make_tensor(OwnTensor::Tensor::randn<float>(Shape{{1,2048}}, OwnTensor::TensorOptions().with_req_grad(true)), "b1");

    auto w2 = ag::make_tensor(OwnTensor::Tensor::randn<float>(Shape{{2048,2048}}, OwnTensor::TensorOptions().with_req_grad(true)), "w2");
    auto b2 = ag::make_tensor(OwnTensor::Tensor::randn<float>(Shape{{1,2048}}, OwnTensor::TensorOptions().with_req_grad(true)), "b2");

    auto w3 = ag::make_tensor(OwnTensor::Tensor::randn<float>(Shape{{2048,2048}}, OwnTensor::TensorOptions().with_req_grad(true)), "w3");
    auto b3 = ag::make_tensor(OwnTensor::Tensor::randn<float>(Shape{{1,2048}}, OwnTensor::TensorOptions().with_req_grad(true)), "b3");

    // Initial inputs (all [1, 2048])
    auto x1 = ag::make_tensor(OwnTensor::Tensor::randn<float>(Shape{{1,2048}}, OwnTensor::TensorOptions().with_req_grad(false)), "x1");
    auto x2 = ag::make_tensor(OwnTensor::Tensor::randn<float>(Shape{{1,2048}}, OwnTensor::TensorOptions().with_req_grad(false)), "x2");
    auto x3 = ag::make_tensor(OwnTensor::Tensor::randn<float>(Shape{{1,2048}}, OwnTensor::TensorOptions().with_req_grad(false)), "x3");

    // Branch 1: 10,000 linear+relu operations
    std::cout << "Building branch 1..." << std::endl;
    auto l1 = x1;
    for (int i = 0; i < 10; i++) {
        l1 = ag::relu(ag::linear(l1, w1, b1));
        if (i % 2000 == 0) std::cout << "  Branch 1: layer " << i << std::endl;
    }
    // Branch 2: 10,000 linear+relu operations
    std::cout << "Building branch 2..." << std::endl;
    auto l2 = x2;
    for (int i = 0; i < 10; i++) {
        l2 = ag::relu(ag::linear(l2, w2, b2));
        if (i % 2000 == 0) std::cout << "  Branch 2: layer " << i << std::endl;
    }
    // Branch 3: 10,000 linear+relu operations
    std::cout << "Building branch 3..." << std::endl;
    auto l3 = x3;
    for (int i = 0; i < 10; i++) {
        l3 = ag::relu(ag::linear(l3, w3, b3));
        if (i % 2000 == 0) std::cout << "  Branch 3: layer " << i << std::endl;
    }
    std::cout << "\nMerging branches..." << std::endl;
    // Merge the three branches
    auto y1 = ag::mul(l1, l2);           // [1,2048] * [1,2048] = [1,2048]
    auto y2 = ag::add(y1, l3);           // [1,2048] + [1,2048] = [1,2048]
    auto y3 = ag::relu(y2);              // [1,2048] -> [1,2048]
    // Loss
    auto target = ag::make_tensor(OwnTensor::Tensor::randn<float>(Shape{{1, 2048}}, OwnTensor::TensorOptions().with_req_grad(false)), "target");
    auto loss = ag::mse_loss(y3, target);

    std::cout << "\n=== Graph built with ~30,000 nodes! ===" << std::endl;
    std::cout << "Starting PARALLEL backward pass..." << std::endl;
    std::cout << "Monitor your system (htop/top) to see all CPU cores active!\n" << std::endl;
    
    ag::backward(loss, nullptr, true);  // enable_parallel = true
    
    std::cout << "\n✅ Backward pass completed!" << std::endl;
    return 0;
}