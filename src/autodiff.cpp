// // =====================
// // autodiff.cpp
// // =====================
// #include <unordered_map>
// #include <cmath>
// #ifndef M_PI
// #define M_PI 3.14159265358979323846
// #endif
// #ifndef SQRT_2_OVER_PI
// #define SQRT_2_OVER_PI 0.7978845608028654f
// #endif
// #include "ad/autodiff.hpp"
// #include "ad/debug.hpp"


// namespace ag {

//     // =========================================================
//     void zero_grad(const Value& root){
//         auto order = topo_from(root.node.get());
//         for(Node* n : order) if(n->requires_grad) n->grad = Tensor::zeros_like(n->value);
//     }
//     // =========================================================
//     // Backpropagation (reverse-mode autodiff)
//     // a.k.a VJP (vector-Jacobian product)
//     void backward(const Value& root, const Tensor* grad_seed){

//         auto order = topo_from(root.node.get());

//         // seed root
//         if(root.node->requires_grad){
//             if(grad_seed) root.node->grad = *grad_seed;
//             else root.node->grad = (root.node->value.size()==1 ? Tensor::ones(1,1) : Tensor::ones_like(root.node->value));
//         }

//         // propagate
//         for(auto it = order.rbegin(); it!=order.rend(); ++it){
//             Node* n = *it; 
//             if(!n->requires_grad) continue; 

//             const Tensor gy = n->grad;
//             ag::debug::on_backprop_step(n, gy);

//             switch(n->op){
//                 case Op::Add: {
//                 Node* A=n->inputs[0].get(), *B=n->inputs[1].get();
//                 if(A->requires_grad) A->grad.add_( Tensor::reduce_to(gy, A->value) );
//                 if(B->requires_grad) B->grad.add_( Tensor::reduce_to(gy, B->value) );
//                 break; }

//                 case Op::Sub: {
//                 Node* A=n->inputs[0].get(), *B=n->inputs[1].get();
//                 if(A->requires_grad) A->grad.add_( Tensor::reduce_to(gy, A->value) );
//                 if(B->requires_grad) B->grad.add_( Tensor::reduce_to(-gy, B->value) );
//                 break; }

//                 case Op::Mul: {
//                 Node* A=n->inputs[0].get(), *B=n->inputs[1].get();
//                 if(A->requires_grad) A->grad.add_( Tensor::reduce_to( gy * B->value, A->value ) );
//                 if(B->requires_grad) B->grad.add_( Tensor::reduce_to( gy * A->value, B->value ) );
//                 break; }

//                 case Op::Relu: {
//                 Node* X=n->inputs[0].get(); 
//                 if(X->requires_grad){ Tensor g = gy * Tensor::relu_mask(n->value); X->grad.add_(g);} 
//                 break; 
//             }

//                 case Op::MatMul: {
//                 Node* A=n->inputs[0].get(), *B=n->inputs[1].get(); 
//                 if(A->requires_grad) A->grad.add_( Tensor::matmul(gy, Tensor::transpose(B->value)) ); 
//                 if(B->requires_grad) B->grad.add_( Tensor::matmul( Tensor::transpose(A->value), gy ) ); 
//                 break; 
//             }
                
//                 case Op::Sum: {
//                 Node* X=n->inputs[0].get(); 
//                 if(X->requires_grad){ float s = gy(0,0); Tensor g = Tensor::ones_like(X->value) * s; X->grad.add_(g);} 
//                 break; 
//             }

//                 case Op::Exp: {
//                 Node* X=n->inputs[0].get(); 
//                 if(X->requires_grad){ Tensor g = gy * Tensor::exp(X->value); X->grad.add_( Tensor::reduce_to(g, X->value) ); } 
//                 break; 
//             }

//                 case Op::Log: {
//                 Node* X=n->inputs[0].get(); 
//                 if(X->requires_grad){ Tensor g = gy / X->value; X->grad.add_( Tensor::reduce_to(g, X->value) ); } 
//                 break; 
//             }


//                 case Op::Tanh: {
//                     Node* X = n->inputs[0].get();
//                     if (X->requires_grad) {
//                         Tensor th  = n->value;                    // tanh(x)
//                         Tensor one = Tensor::ones_like(th);
//                         Tensor g   = gy * (one - th * th);        // gy * (1 - tanh^2)
//                         X->grad.add_( Tensor::reduce_to(g, X->value) );
//                     }
//                     break;
//                 }

//                 case Op::Sigmoid:{
//                     Node* X=n->inputs[0].get(); 
//                     if(X->requires_grad){ 
                        
//                         Tensor s = Tensor::sigmoid(X->value); 
//                         Tensor g = gy * ( s * (Tensor::ones_like(s) - s) ); 
//                         X->grad.add_( Tensor::reduce_to(g, X->value) ); 
//                     } 
//                     break; 
//                 }
                    
//                 case Op::Softplus:{
//                     Node* X=n->inputs[0].get(); 
//                     if(X->requires_grad){ 
//                         Tensor g = gy * Tensor::sigmoid(X->value); 
//                         X->grad.add_( Tensor::reduce_to(g, X->value) ); 
//                     } 
//                     break; 
//                 }
                
//                 case Op::SiLU:{
//                     Node* X=n->inputs[0].get(); 
//                     if(X->requires_grad){ 
//                         Tensor s = Tensor::sigmoid(X->value); 
//                         Tensor g = gy * ( s + X->value * ( s * (Tensor::ones_like(s) - s) ) ); 
//                         X->grad.add_( Tensor::reduce_to(g, X->value) ); 
//                     } 
//                     break; 
//                 }

//                 case Op::GELU: { // tanh approximation derivative
//                     Node* X = n->inputs[0].get();
//                     if (X->requires_grad) {
//                         const float c = SQRT_2_OVER_PI;
//                         int R = X->value.rows(), C = X->value.cols();
//                         Tensor x = X->value, u(R,C), dudx(R,C);
//                         for (int i=0;i<R;++i) for (int j=0;j<C;++j) {
//                             float z = x(i,j);
//                             u(i,j)    = c * ( z + 0.044715f * z*z*z );
//                             dudx(i,j) = c * ( 1.f + 0.134145f * z*z );
//                         }
//                         Tensor th   = Tensor::tanh(u);
//                         Tensor one  = Tensor::ones_like(th);
//                         Tensor dgelu = (one + th) * 0.5f
//                                     + (x * ((one - th*th) * dudx)) * 0.5f;
//                         Tensor g = gy * dgelu;
//                         X->grad.add_( Tensor::reduce_to(g, X->value) );
//                     }
//                     break;
//                 }

//                 case Op::LeakyRelu: {
//                     Node* X = n->inputs[0].get();
//                     Node* A = n->inputs[1].get();  // alpha as [1,1]
//                     float a = A->value(0,0);
//                     if (X->requires_grad) {
//                         int R = X->value.rows(), C = X->value.cols();
//                         Tensor g(R,C);
//                         for (int i=0;i<R;++i) for (int j=0;j<C;++j) {
//                             float z = X->value(i,j);
//                             g(i,j) = gy(i,j) * (z > 0.f ? 1.f : a);
//                         }
//                         X->grad.add_( g );
//                     }
//                     // (no grad to alpha — treat as hyperparam)
//                     break;
//                 }

//                 case Op::RowSum:{ 
//                     Node* X=n->inputs[0].get(); 
//                     if(X->requires_grad){ 
//                         Tensor g = gy * Tensor::ones_like(X->value); 
//                         X->grad.add_(g); 
//                     } 
//                     break; 
//                 }

//                 case Op::RowMax: {
//                     Node* X = n->inputs[0].get();
//                     if (X->requires_grad) {
//                         int R = X->value.rows(), C = X->value.cols();
//                         Tensor m = Tensor::row_max(X->value);
//                         Tensor g(R,C);
//                         for (int i=0;i<R;++i) for (int j=0;j<C;++j)
//                             g(i,j) = (X->value(i,j) == m(i,0)) ? gy(i,0) : 0.f;
//                         X->grad.add_( g );
//                     }
//                     break;
//                 }

//                 case Op::MeanAll:{
//                     Node* X=n->inputs[0].get(); 
//                     if(X->requires_grad){ 
                        
//                         float s = gy(0,0)/(float)(X->value.rows() * X->value.cols()); 
//                         Tensor g = Tensor::ones_like(X->value) * s; 
//                         X->grad.add_(g); 
//                     } 
//                     break; 
//                 }

//                 case Op::SoftmaxRow:{ 
//                     Node* Z=n->inputs[0].get(); 
//                     if(Z->requires_grad){ 
//                         Tensor y = n->value; Tensor dot = Tensor::row_sum( y * gy ); Tensor g = y * ( gy - dot ); 
//                         Z->grad.add_(g); 
//                     } 
//                     break; 
//                 }

//                 case Op::LogSumExpRow: {
//                     Node* Z = n->inputs[0].get();
//                     if (Z->requires_grad) {
//                         Tensor y = Tensor::softmax_row(Z->value); // d lse / dz = softmax(z)
//                         Tensor g = y * gy;                        // broadcast [B,1] over cols
//                         Z->grad.add_( g );
//                     }
//                     break;
//                 }

//                 case Op::CeWithLogits:{ 
//                     Node* Z=n->inputs[0].get(); 
//                     Node* Y=n->inputs[1].get(); 
//                     int B = Z->value.rows();
//                     // dL/dZ = (softmax(Z) - Y)/B ; dL/dY = -log_softmax(Z)/B
//                     Tensor sm = Tensor::softmax_row(Z->value);
//                     Tensor gZ = (sm - Y->value) * (1.0f / (float)B);
//                     if(Z->requires_grad) Z->grad.add_(gZ);
//                     if(Y->requires_grad){ Tensor lse = Tensor::logsumexp_row(Z->value); Tensor lsm = Z->value - lse; Tensor gY = (lsm * (-1.0f/(float)B)); Y->grad.add_(gY); }
//                     break; }
//                     case Op::Leaf: default: break;
//             }
//         }
//     }

//     // =========================================================
//     // Forward-mode autodiff (Jacobian-vector product)
//     // a.k.a JVP (Jacobian-vector product)

//     Tensor jvp(const Value& root, const std::unordered_map<Node*, Tensor>& seed){

//         auto order = topo_from(root.node.get());
//         std::unordered_map<Node*, Tensor> T; T.reserve(order.size());


//         auto zero_like = [](const Tensor& x){ 
//             return Tensor::zeros_like(x); 
//         };


//         for(Node* n : order){
//             Tensor t = seed.count(n) ? seed.at(n) : zero_like(n->value); // seed or zero
            
//             ag::debug::on_jvp_step(n);  // for debugging
            
//             switch(n->op){
//             case Op::Leaf: break; // t already the seed
            
//             case Op::Add: { 
//                 Node* A=n->inputs[0].get(), *B=n->inputs[1].get(); t = T[A] + T[B]; 
//                 break; 
//             }
            
//             case Op::Sub: { 
//                 Node* A=n->inputs[0].get(), *B=n->inputs[1].get(); t = T[A] + (-T[B]); 
//                 break; 
//             }
            
//             case Op::Mul: { 
//                 Node* A=n->inputs[0].get(), *B=n->inputs[1].get(); t = (T[A] * B->value) + (A->value * T[B]); 
//                 break; 
//             }
            
//             case Op::Relu:{ 
//                 Node* X=n->inputs[0].get(); t = T[X] * Tensor::relu_mask(n->value); 
//                 break; 
//             }
            
//             case Op::MatMul:{ 
//                 Node* A=n->inputs[0].get(), *B=n->inputs[1].get(); t = Tensor::matmul(T[A], B->value); t = t + Tensor::matmul(A->value, T[B]); 
//                 break; 
//             }
            
//             case Op::Sum: { 
//                 Node* X=n->inputs[0].get(); float s = T[X].sum_scalar(); t = Tensor(1,1); t(0,0)=s; break; 
//             }
            
//             case Op::Exp:{ 
//                 Node* X=n->inputs[0].get(); t = T[X] * Tensor::exp(X->value); 
//                 break; 
//             }
            
//             case Op::Log:{ 
//                 Node* X=n->inputs[0].get(); t = T[X] / X->value; 
//                 break; 
//             }

//             case Op::Tanh: {
//                 Node* X = n->inputs[0].get();


//                 Tensor th  = n->value;
//                 Tensor one = Tensor::ones_like(th);
//                 t = T[X] * (one - th * th);
//                 break;
//             }

//             case Op::Sigmoid:{ 
//                 Node* X=n->inputs[0].get(); Tensor s = Tensor::sigmoid(X->value); t = T[X] * ( s * (Tensor::ones_like(s) - s) ); 
//                 break; 
//             }
            
//             case Op::Softplus:{ 
//                 Node* X=n->inputs[0].get(); t = T[X] * Tensor::sigmoid(X->value); 
//                 break; 
//             }
            
//             case Op::SiLU:{ 
//                 Node* X=n->inputs[0].get(); Tensor s = Tensor::sigmoid(X->value); t = T[X] * ( s + X->value * ( s * (Tensor::ones_like(s) - s) ) ); 
//                 break; 
//             }

//             case Op::GELU: {
//                 Node* X = n->inputs[0].get();
//                 const float c = SQRT_2_OVER_PI;
//                 int R = X->value.rows(), C = X->value.cols();
//                 Tensor x = X->value, u(R,C), dudx(R,C);
//                 for (int i=0;i<R;++i) for (int j=0;j<C;++j) {
//                     float z = x(i,j);
//                     u(i,j)    = c * ( z + 0.044715f * z*z*z );
//                     dudx(i,j) = c * ( 1.f + 0.134145f * z*z );
//                 }
//                 Tensor th   = Tensor::tanh(u);
//                 Tensor one  = Tensor::ones_like(th);
//                 Tensor dgelu = (one + th) * 0.5f
//                             + (x * ((one - th*th) * dudx)) * 0.5f;
//                 t = T[X] * dgelu;
//                 break;
//             }

//             case Op::LeakyRelu: {
//                 Node* X = n->inputs[0].get();
//                 Node* A = n->inputs[1].get();
//                 float a = A->value(0,0);
//                 int R = X->value.rows(), C = X->value.cols();
//                 t = Tensor::zeros(R,C);
//                 for (int i=0;i<R;++i) for (int j=0;j<C;++j) {
//                     float z = X->value(i,j);
//                     t(i,j) = T[X](i,j) * (z > 0.f ? 1.f : a);
//                 }
//                 break;
//             }

//             case Op::RowSum:{ Node* X=n->inputs[0].get(); t = Tensor::row_sum(T[X]); break; }

//             case Op::RowMax: { // subgradient: pass only along argmax entries
//                 Node* X = n->inputs[0].get();
//                 int R = X->value.rows(), C = X->value.cols();
//                 Tensor m = Tensor::row_max(X->value);
//                 Tensor M(R,C);
//                 for (int i=0;i<R;++i) for (int j=0;j<C;++j)
//                     M(i,j) = (X->value(i,j) == m(i,0)) ? 1.f : 0.f;
//                 t = Tensor::row_sum( T[X] * M );
//                 break;
//             }

//             case Op::MeanAll: {
//                 Node* X = n->inputs[0].get();
//                 float s = 1.f / float(X->value.rows() * X->value.cols());
//                 t = Tensor(1,1);
//                 t(0,0) = T[X].sum_scalar() * s;
//                 break;
//             }

//             case Op::SoftmaxRow:{ 
                
//                 Node* Z=n->inputs[0].get(); Tensor y = n->value; Tensor dot = Tensor::row_sum( y * T[Z] ); t = y * ( T[Z] - dot ); 
                
//                 break; 
            
//             }

//             case Op::LogSumExpRow: {
//                 Node* Z = n->inputs[0].get();
//                 Tensor y = Tensor::softmax_row(Z->value);
//                 t = Tensor::row_sum( y * T[Z] );
//                 break;
//             }

//             case Op::CeWithLogits:{ 
//                 Node* Z=n->inputs[0].get(); 
//                 Node* Y=n->inputs[1].get(); int B=Z->value.rows(); 
//                 Tensor sm = Tensor::softmax_row(Z->value); 
//                 Tensor gZ = (sm - Y->value) * (1.0f/(float)B); 
//                 Tensor tZ = T.count(Z)? T[Z] : Tensor::zeros_like(Z->value);                 
//                 Tensor tY = T.count(Y)? T[Y] : Tensor::zeros_like(Y->value); 
//                 Tensor lse = Tensor::logsumexp_row(Z->value); 
//                 Tensor lsm = Z->value - lse; Tensor gY = (lsm * (-1.0f/(float)B)); 
//                 Tensor dotZ(1,1); dotZ(0,0) = (gZ * tZ).sum_scalar(); 
//                 Tensor dotY(1,1); dotY(0,0) = (gY * tY).sum_scalar(); 
//                 t = dotZ + dotY; break; }
//             }
//             T[n] = t;
//         }
//         return T[root.node.get()];
//     }


// } // namespace ag

// src/autodiff.cpp
#include <unordered_map>
#include "ad/autodiff.hpp"
#include "ad/autodiff_ops.hpp"
#include "ad/debug.hpp"

namespace ag {

void zero_grad(const Value& root){
    auto order = topo_from(root.node.get());
    for (Node* n : order) if (n->requires_grad) n->grad = Tensor::zeros_like(n->value);
}

void backward(const Value& root, const Tensor* grad_seed){
    auto order = topo_from(root.node.get());

    // seed
    if (root.node->requires_grad) {
        root.node->grad = grad_seed ? *grad_seed
                                    : (root.node->value.size()==1 ? Tensor::ones(1,1)
                                                                  : Tensor::ones_like(root.node->value));
    }

    // reverse topo
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node* n = *it;
        if (!n->requires_grad) continue;
        const Tensor& gy = n->grad;

        ag::debug::on_backprop_step(n, gy); // (optional) prints one line per node

        VjpFn fn = vjp_lookup(n->op);
        if (fn) fn(n, gy); // handler accumulates into parents
    }
}

Tensor jvp(const Value& root, const std::unordered_map<Node*, Tensor>& seed){
    auto order = topo_from(root.node.get());
    std::unordered_map<Node*, Tensor> T;
    T.reserve(order.size());

    auto tangent_of = [&](Node* p) -> const Tensor& {
        auto it = T.find(p);
        if (it != T.end()) return it->second;
        static Tensor Z; // fallback; shouldn't be used for leaves without seeds
        return Z;
    };

    for (Node* n : order) {
        // seed tangent for this node (if provided), else zeros
        Tensor t = Tensor::zeros_like(n->value);
        if (auto it = seed.find(n); it != seed.end()) t = it->second;

        ag::debug::on_jvp_step(n); // (optional) prints forward-mode step

        JvpFn fn = jvp_lookup(n->op);
        if (fn) t = fn(n, tangent_of);

        T[n] = t;
    }
    return T[root.node.get()];
}

} // namespace ag
