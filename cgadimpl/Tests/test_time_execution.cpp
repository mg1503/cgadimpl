#include "ad/ag_all.hpp"


int main(){
    OwnTensor::Tensor A = OwnTensor::Tensor::randn<float>(OwnTensor::Shape{{1024,1024}}, OwnTensor::TensorOptions().with_req_grad(true));
    OwnTensor::Tensor B =  OwnTensor::Tensor::randn<float>(OwnTensor::Shape{{1024,1024}}, OwnTensor::TensorOptions().with_req_grad(true));

    auto a = ag::make_tensor(A, "a");
    auto b = ag::make_tensor(B, "b");

    auto c = ag::matmul(a,b);

    return 0;
}