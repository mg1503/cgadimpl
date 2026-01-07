#include<iostream>
#include "ad/ag_all.hpp"
int main(){
    Tensor J_tensor=Tensor::randn<float>(Shape{{2,2}},TensorOptions().with_req_grad(true));
    Tensor G_tensor=Tensor::randn<float>(Shape{{2,2}},TensorOptions().with_req_grad(true));
    Tensor R_tensor=Tensor::randn<float>(Shape{{2,2}},TensorOptions().with_req_grad(true));
    Tensor V_tensor=Tensor::randn<float>(Shape{{2,2}},TensorOptions().with_req_grad(true));
    Tensor K_tensor=Tensor::randn<float>(Shape{{2,2}},TensorOptions().with_req_grad(true));
   
    auto j=ag::make_tensor(J_tensor,"J");
    auto g=ag::make_tensor(G_tensor,"G");
    auto r=ag::make_tensor(R_tensor,"R");
    auto v=ag::make_tensor(V_tensor,"V");
    auto k=ag::make_tensor(K_tensor,"K");

    auto y_true = ag::make_tensor(Tensor::randn<float>(Shape{{2,2}}, TensorOptions().with_req_grad(true)), "y_true");

    for(int i=0;i<2;i++){

        auto jg=ag::matmul(j,g)+r;
        auto jk=ag::matmul(g,j)+k;
        auto gj=ag::relu(jk)+v;
        auto a= ag::relu(gj)+jg;
        auto l = ag::mse_loss(a, y_true);

        ag::debug::print_value("Loss (l)", l);
        ag::zero_grad(l);
        ag::debug::dump_dot(l, "graph_jgrvk.jpg");
        ag::backward(l, nullptr, true);
    
        ag::debug::print_grad("Grad of j",j);
        ag::debug::print_grad("Grad of g",g);
        ag::debug::print_grad("Grad of r",r);

    }
    return 0;

}