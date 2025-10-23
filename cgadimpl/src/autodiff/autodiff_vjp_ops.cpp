// // ============================================
// // cgadimpl/src/autodiff/autodiff_vjp_ops.cpp
// // ============================================

// #include "ad/detail/autodiff_ops.hpp"
// #include "ad/runtime.hpp"
// #include <cmath>
// #include "ad/nodeops.hpp"

// namespace ag {
// namespace detail{

// // helper: reduce a gradient to a parent's shape (broadcast-aware)
// inline Tensor rt(const Tensor& g, const Tensor& like){ return Tensor::reduce_to(g, like); }

// // ----- elementwise binary -----
// void vjp_Add(Node* n, const Tensor& gy){
//     Node* A = n->inputs[0].get();
//     Node* B = n->inputs[1].get();

//     // MODIFIED: Add device dispatch
//     if (A->value.is_cpu()) {
//         if (A->requires_grad) A->grad.add_( rt(gy, A->value) );
//         if (B->requires_grad) B->grad.add_( rt(gy, B->value) );
//     } else {
//         // Assumes grads have been pre-allocated on the correct device
//         ag::kernels::cuda().vjp_add(A->grad.data(), B->grad.data(), gy.data(),
//                                     gy.numel(), ag::current_stream());
//     }
// }
// void vjp_Sub(Node* n, const Tensor& gy){
//     Node* A = n->inputs[0].get(); Node* B = n->inputs[1].get();
//     if (A->requires_grad) A->grad.add_( rt(gy, A->value) );
//     if (B->requires_grad) B->grad.add_( rt(-gy, B->value) );
// }
// void vjp_Mul(Node* n, const Tensor& gy){
//     Node* A = n->inputs[0].get(); Node* B = n->inputs[1].get();
//     if (A->requires_grad) A->grad.add_( rt( gy * B->value, A->value) );
//     if (B->requires_grad) B->grad.add_( rt( gy * A->value, B->value) );
// }

// // ----- elementwise trinary -----
// void vjp_FMA(Node* n, const Tensor& gy){
//     Node* A = n->inputs[0].get();
//     Node* B = n->inputs[1].get();
//     Node* C = n->inputs[2].get();

//     // External kernel (if plugin loaded), else fallback to Tensor::matmul
//     auto* mm = ag::kernels::cpu().matmul;

//     // Shapes
//     const Tensor& At = A->value;
//     const Tensor& Bt = B->value;
//     auto [M, K]  = At.shape();
//     auto [K2, N] = Bt.shape();
//     (void)K2; // assume forward already checked

//     if (A->requires_grad){
//         Tensor BT = Tensor::transpose(Bt); // (N x K)
//         Tensor dA(M, K);                   // temp buffer

//         if (mm) {
//             // dA = gy (MxN) * BT (NxK)
//             mm(gy.data(), BT.data(), dA.data(), M, N, K);
//         } else {
//             dA = Tensor::matmul(gy, BT);
//         }
//         A->grad.add_(dA);
//     }

//     if (B->requires_grad){
//         Tensor AT = Tensor::transpose(At); // (K x M)
//         Tensor dB(K, N);                   // temp buffer

//         if (mm) {
//             // dB = AT (KxM) * gy (MxN)
//             mm(AT.data(), gy.data(), dB.data(), K, M, N);
//         } else {
//             dB = Tensor::matmul(AT, gy);
//         }
//         B->grad.add_(dB);
//     }
//     if (C->requires_grad) C->grad.add_( rt(gy, C->value) );
// }

// void vjp_LayerNorm(Node* n, const Tensor& gy){

//     Node* x = n->inputs[0].get();
//      int N = x->value.cols(); // normalize over last dim (row-wise)

//    //  debug::print_tensor("gy",gy);
     
    
//     // stddev [2x1] - float
//     Tensor std = Tensor::sqrt(*(n->tape[0]) +0.01);
//   //  debug::print_tensor("std",std);

//     // x - mean [2x1]
//     Tensor xmu = x->value - *(n->tape[1]);
//  //   debug::print_tensor("xmu",xmu);

//     // sum of grad_out along row
//     Tensor grad_sum = Tensor::row_sum(gy);
//   //  debug::print_tensor("grad_sum",grad_sum);

//     // dot(grad_out, x - mean) along row
//     Tensor grad_dot_xmu = Tensor::row_sum(gy * xmu);
//    // debug::print_tensor("grad_dot_xmu",grad_dot_xmu);

//     // term: N * grad_out
//     Tensor term1 = gy * float(N);
//  //   debug::print_tensor("term1",term1);

//     // term: subtract sum of grad_out
//     Tensor term2 = term1 - grad_sum;
//    // debug::print_tensor("term2",term2);

//     // term: subtract (x - mean) * (grad_dot_xmu / (var + eps))
//     Tensor term3 = term2 - (xmu * (grad_dot_xmu / (*(n->tape[0]) + 0.01)));
//   //  debug::print_tensor("term3",term3);

//     // scale: divide by (N * std)
//     Tensor dx = term3 / (std * float(N));
//   //  debug::print_tensor("dx",dx);

//     if (x->requires_grad) x->grad.add_( dx );

// }


// void vjp_RealLayerNorm(Node* n, const Tensor& gy){

//     Node* x = n->inputs[0].get();
//     Node* b = n->inputs[1].get();
//     Node* g = n->inputs[2].get();
//      int N = x->value.cols(); // normalize over last dim (row-wise)

//    //  debug::print_tensor("gy",gy);
     
    
//     // stddev [2x1] - float
//     Tensor std = Tensor::sqrt(*(n->tape[0]) +0.01);
//   //  debug::print_tensor("std",std);

//     // x - mean [2x1]
//     Tensor xmu = x->value - *(n->tape[1]);
//  //   debug::print_tensor("xmu",xmu);

//     // sum of grad_out along row
//     Tensor grad_sum = Tensor::row_sum(gy);
//   //  debug::print_tensor("grad_sum",grad_sum);

//     // dot(grad_out, x - mean) along row
//     Tensor grad_dot_xmu = Tensor::row_sum(gy * xmu);
//    // debug::print_tensor("grad_dot_xmu",grad_dot_xmu);

//     // term: N * grad_out
//     Tensor term1 = gy * float(N);
//  //   debug::print_tensor("term1",term1);

//     // term: subtract sum of grad_out
//     Tensor term2 = term1 - grad_sum;
//    // debug::print_tensor("term2",term2);

//     // term: subtract (x - mean) * (grad_dot_xmu / (var + eps))
//     Tensor term3 = term2 - (xmu * (grad_dot_xmu / (*(n->tape[0]) + 0.01)));
//   //  debug::print_tensor("term3",term3);

//     // scale: divide by (N * std)
//     Tensor dx = term3 / (std * float(N));
//  //debug::print_tensor("dx",term3);
// // debug::print_tensor("g",g->value);

//     if (x->requires_grad) x->grad.add_( dx);
// if (b->requires_grad) b->grad.add_( Tensor::row_sum(gy) );   // db = sum over batch
// if (g->requires_grad) g->grad.add_( Tensor::row_sum(gy * (*(n->tape[2]))) );

// }


// void vjp_RMSNorm(Node* n, const Tensor& gy){

//     Node* x = n->inputs[0].get();
//     Tensor rms = *n->tape[0];
//     Tensor y   = *n->tape[1];   // normalized x

//     // upstream dot
//     Tensor dot = Tensor::row_sum(gy * y);  // [batch x 1]

//     Tensor grad_x = (gy / rms) - (y * dot / (rms*x->value.cols()));

//     if (x->requires_grad) x->grad.add_(grad_x);


// }


// void vjp_RealRMSNorm(Node* n, const Tensor& gy){

//     Node* x = n->inputs[0].get();
//     Node* g = n->inputs[1].get();
//     Tensor rms = *n->tape[0];
//     Tensor y   = *n->tape[1];   // normalized x


//     // upstream dot
//     Tensor dot = Tensor::row_sum(gy * y);  // [batch x 1]

//     Tensor grad_x = g->value*((gy / rms) - (y * dot / (rms*x->value.cols())));


//     if (x->requires_grad) x->grad.add_(grad_x);
//     if (g->requires_grad) g->grad.add_( gy * (x->value / rms));


// }


// // ----- elementwise quarternary -----
// void vjp_Attention(Node* n, const Tensor& gy){
//     Node* A = n->inputs[0].get();
//     Node* B = n->inputs[1].get();
//     Node* C = n->inputs[2].get();
//     Node* D = n->inputs[3].get();
    
//     Tensor q = n->tape[0] ? *n->tape[0] : Tensor();
//     Tensor k = n->tape[1] ? *n->tape[1] : Tensor();
//     Tensor v = n->tape[2] ? *n->tape[2] : Tensor();
//     float scale = 1.0f / std::sqrt(float(k.cols()));
//     Tensor s = n->tape[3] ? *n->tape[3] : Tensor();

//     // ---- Backprop chain ----

//     // y = s v
//     Tensor dL_ds = Tensor::matmul(gy, Tensor::transpose(v));   // [B x B]
//     Tensor dL_dv = Tensor::matmul(Tensor::transpose(s), gy);   // [A x D]

//     // s = softmax(g)
//     Tensor dL_dg; 
//     {
//         Tensor dot = Tensor::row_sum(s * dL_ds);
//         dL_dg = s * (dL_ds - dot);
//     }

//     // g = q k^T
//     Tensor dL_dq = Tensor::matmul(dL_dg, k);
//     Tensor dL_dk = Tensor::matmul(Tensor::transpose(dL_dg), q);

//     // q = A B
//     Tensor dL_dA_q = Tensor::matmul(dL_dq, Tensor::transpose(B->value));
//     Tensor dL_dB   = Tensor::matmul(Tensor::transpose(A->value), dL_dq)* scale;;

//     // k = A C
//     Tensor dL_dA_k = Tensor::matmul(dL_dk, Tensor::transpose(C->value));
//     Tensor dL_dC   = Tensor::matmul(Tensor::transpose(A->value), dL_dk)* scale;

//     // v = A D
//     Tensor dL_dA_v = Tensor::matmul(dL_dv, Tensor::transpose(D->value));
//     Tensor dL_dD   = Tensor::matmul(Tensor::transpose(A->value), dL_dv);

//     // combine A contributions
//     Tensor dL_dA = dL_dA_q + dL_dA_k + dL_dA_v;

//     // ---- Accumulate ----
//     if (A->requires_grad) A->grad.add_(dL_dA);
//     if (B->requires_grad) B->grad.add_(dL_dB);
//     if (C->requires_grad) C->grad.add_(dL_dC);
//     if (D->requires_grad) D->grad.add_(dL_dD);

// }


// void vjp_AlibiAttention(Node* n, const Tensor& gy){
//     Node* A = n->inputs[0].get();
//     Node* B = n->inputs[1].get();
//     Node* C = n->inputs[2].get();
//     Node* D = n->inputs[3].get();
    
//     Tensor q = n->tape[0] ? *n->tape[0] : Tensor();
//     Tensor k = n->tape[1] ? *n->tape[1] : Tensor();
//     Tensor v = n->tape[2] ? *n->tape[2] : Tensor();
//     float scale = 1.0f / std::sqrt(float(k.cols()));
//     Tensor s = n->tape[3] ? *n->tape[3] : Tensor();

//     // ---- Backprop chain ----

//     // y = s v
//     Tensor dL_ds = Tensor::matmul(gy, Tensor::transpose(v));   // [B x B]
//     Tensor dL_dv = Tensor::matmul(Tensor::transpose(s), gy);   // [A x D]

//     // s = softmax(g)
//     Tensor dL_dg; 
//     {
//         Tensor dot = Tensor::row_sum(s * dL_ds);
//         dL_dg = s * (dL_ds - dot);
//     }

//     // g = q k^T
//     Tensor dL_dq = Tensor::matmul(dL_dg, k);
//     Tensor dL_dk = Tensor::matmul(Tensor::transpose(dL_dg), q);

//     // q = A B
//     Tensor dL_dA_q = Tensor::matmul(dL_dq, Tensor::transpose(B->value));
//     Tensor dL_dB   = Tensor::matmul(Tensor::transpose(A->value), dL_dq)* scale;;

//     // k = A C
//     Tensor dL_dA_k = Tensor::matmul(dL_dk, Tensor::transpose(C->value));
//     Tensor dL_dC   = Tensor::matmul(Tensor::transpose(A->value), dL_dk)* scale;

//     // v = A D
//     Tensor dL_dA_v = Tensor::matmul(dL_dv, Tensor::transpose(D->value));
//     Tensor dL_dD   = Tensor::matmul(Tensor::transpose(A->value), dL_dv);

//     // combine A contributions
//     Tensor dL_dA = dL_dA_q + dL_dA_k + dL_dA_v;

//     // ---- Accumulate ----
//     if (A->requires_grad) A->grad.add_(dL_dA);
//     if (B->requires_grad) B->grad.add_(dL_dB);
//     if (C->requires_grad) C->grad.add_(dL_dC);
//     if (D->requires_grad) D->grad.add_(dL_dD);

// }


// void vjp_SWIGLU(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     Node* A = n->inputs[1].get();
//     Node* B = n->inputs[2].get();
//     Node* C = n->inputs[3].get();
//     Node* D = n->inputs[4].get();

//     Tensor y = Tensor::matmul(X->value, Tensor::transpose(A->value)) + B->value;
//     Tensor q = y * Tensor::sigmoid(y);
//     Tensor h = Tensor::matmul(X->value, Tensor::transpose(C->value)) + D->value;
//     Tensor w = q * h;

//     // derivatives
//     Tensor Swishdif = Tensor::sigmoid(y) + y * (Tensor::sigmoid(y) * (Tensor::ones_like(y) - Tensor::sigmoid(y)));

//     Tensor dL_dB = Swishdif * h * gy;
//     Tensor dL_dA = Tensor::matmul(Tensor::transpose(Swishdif * h * gy), X->value);

//     Tensor dL_dD = q*gy;
//     Tensor dL_dC = Tensor::matmul(Tensor::transpose(q * gy), X->value);

//     Tensor dL_dX = Tensor::matmul(Swishdif * h * gy, A->value)
//                  + Tensor::matmul(q * gy, C->value);

//     // accumulate grads
//     if (X->requires_grad) X->grad.add_(dL_dX);
//     if (A->requires_grad) A->grad.add_(dL_dA);
//     if (B->requires_grad) B->grad.add_(dL_dB);
//     if (C->requires_grad) C->grad.add_(dL_dC);
//     if (D->requires_grad) D->grad.add_(dL_dD);

// }


// // ----- unary activations -----
// // void vjp_Relu(Node* n, const Tensor& gy){
// //     Node* X = n->inputs[0].get();
// //     if (!X->requires_grad) return;

// //         auto* mm = ag::kernels::cpu().relumask;
// //             auto [K2, N] = (X->value).shape();

// //                 Tensor dA(K2, N);                   // temp buffer

// //         if(mm)
// //             mm((X->value).data(), dA.data(), dA.numel());

// //         else

// //         {
// //             std::cout<<"CUDA is unused";
// //     dA = Tensor::relu_mask(X->value);

// //         }


// //     X->grad.add_( rt( gy * dA, X->value) );

// // }


// void vjp_Relu(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (!X->requires_grad) {
//         return;
//     }

//     // Direct implementation of the ReLU derivative calculation.
//     // The derivative is 1 for positive inputs and 0 for non-positive inputs.
//     // This is calculated by creating a "ReLU mask" from the original input tensor.
//     // The original input is X->value. Note: We use the input X->value, not the output n->value.
//     Tensor dA = Tensor::relu_mask(X->value);

//     // Apply the chain rule: multiply the incoming gradient (gy) by the local derivative (dA)
//     // and add the result to the gradient of the input node X.
//     X->grad.add_( rt( gy * dA, X->value) );
// }

// // void vjp_Exp(Node* n, const Tensor& gy){
// //     Node* X = n->inputs[0].get();
// //     if (!X->requires_grad) return;

// //         auto* mm = ag::kernels::cpu().exp;
// //             auto [K2, N] = (X->value).shape();

// //                 Tensor dA(K2, N);                   // temp buffer

// //         if(mm)
// //             mm((X->value).data(), dA.data(), dA.numel());

// //         else

// //         {
// //             std::cout<<"CUDA is unused";
// //     dA = Tensor::relu_mask(X->value);

// //         }


// //     X->grad.add_( rt( gy * dA, X->value) );
// // }
// void vjp_Exp(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (!X->requires_grad) {
//         return;
//     }

//     // Direct implementation of the exponential derivative calculation.
//     // The derivative of exp(x) is exp(x).
//     // The result of the forward pass, exp(X->value), is already stored in n->value.
//     // Therefore, the local derivative (dA) is simply the node's own value.
//     const Tensor& dA = n->value;

//     // Apply the chain rule: multiply the incoming gradient (gy) by the local derivative (dA)
//     // and add the result to the gradient of the input node X.
//     X->grad.add_( rt( gy * dA, X->value) );
// }


// void vjp_Log(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (X->requires_grad) X->grad.add_( rt( gy / X->value, X->value) );
// }

// void vjp_GCU(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (X->requires_grad) X->grad.add_( rt( gy * (Tensor::cos(X->value)-(X->value*Tensor::sin(X->value))), X->value) );
// }

// void vjp_Mish(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (X->requires_grad) X->grad.add_( rt( gy * (Tensor::tanh( Tensor::softplus(X->value) )-(  (X->value*Tensor::sigmoid(X->value))  / (Tensor::cosh( Tensor::softplus(X->value)*Tensor::cosh( Tensor::softplus(X->value) ))    )            )), X->value) );
// }


// void vjp_Tanh(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (!X->requires_grad) return;
//     Tensor th = n->value, one = Tensor::ones_like(th);
    
//     X->grad.add_( rt( gy * (one - th*th), X->value) );
// }


// // void vjp_Sigmoid(Node* n, const Tensor& gy){

// //  Node* X = n->inputs[0].get();
// //     if (!X->requires_grad) return;

// //         auto* mm = ag::kernels::cpu().sigmoidiff;
// //             auto [K2, N] = (X->value).shape();

// //                 Tensor dA(K2, N);                   // temp buffer

// //         if(mm)
// //             mm((X->value).data(), dA.data(), dA.numel());

// //         else

// //         {
// //             std::cout<<"CUDA is unused";
// //              Tensor s = Tensor::sigmoid(X->value);

// //     dA = ( s * (Tensor::ones_like(s)-s));

// //         }

// //     X->grad.add_( rt( gy * dA, X->value) );
// // }

// void vjp_Sigmoid(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (!X->requires_grad) {
//         return;
//     }

//     // Direct implementation of the sigmoid derivative calculation.
//     // The derivative of sigmoid(x) is y * (1 - y), where y is the output of the sigmoid.
//     // The output 'y' is already stored in the node's value from the forward pass.
//     const Tensor& Y = n->value; // This is sigmoid(X->value)
//     Tensor dA = Y * (Tensor::ones_like(Y) - Y);

//     // Chain rule: multiply the incoming gradient (gy) by the local derivative (dA)
//     // and add it to the gradient of the input node X.
//     X->grad.add_( rt( gy * dA, X->value) );
// }

// void vjp_Softplus(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get(); if (!X->requires_grad) return;
//     X->grad.add_( rt( gy * Tensor::sigmoid(X->value), X->value) );
// }

// void vjp_Gaus(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get(); if (!X->requires_grad) return;
//     X->grad.add_( rt( gy * -2*X->value*Tensor::exp(-1*X->value*X->value), X->value) );
// }

// void vjp_Transpose(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get(); if (!X->requires_grad) return;

//     X->grad.add_( rt( Tensor::transpose(gy) , X->value) );
// }



// void vjp_SiLU(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get(); if (!X->requires_grad) return;
//     Tensor s = Tensor::sigmoid(X->value);
//     X->grad.add_( rt( gy * ( s + X->value * ( s * (Tensor::ones_like(s)-s) ) ), X->value) );
// }

// void vjp_Parcon(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get(); if (!X->requires_grad) return;
//     X->grad.add_( rt( gy * ( 2 *Tensor::ones_like(X->value)- 2*X->value  ), X->value) );
// }

// void vjp_LiSHT(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get(); if (!X->requires_grad) return;
//     X->grad.add_( rt( gy * ( Tensor::tanh(X->value)+ (Tensor::sech(X->value)*Tensor::sech(X->value)*X->value ) ), X->value) );
// }




// void vjp_GELU(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get(); if (!X->requires_grad) return;
//     constexpr float c = 0.79788456080286535588f; // sqrt(2/pi)
//     int R=X->value.rows(), C=X->value.cols();
//     Tensor x=X->value,u(R,C),dudx(R,C);
//     for(int i=0;i<R;++i) for(int j=0;j<C;++j){
//         float z=x(i,j);
//         u(i,j)=c*(z+0.044715f*z*z*z);
//         dudx(i,j)=c*(1.f+0.134145f*z*z);
//     }
//     Tensor th=Tensor::tanh(u), one=Tensor::ones_like(th);
//     Tensor dgelu=(one+th)*0.5f + (x * ((one - th*th) * dudx))*0.5f;
//     X->grad.add_( rt( gy * dgelu, X->value) );
// }
// void vjp_LeakyRelu(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get(); Node* A = n->inputs[1].get();
//     if (!X->requires_grad) return;
//     float a = A->value(0,0);
//     int R=X->value.rows(), C=X->value.cols();
//     Tensor g(R,C);
//     for(int i=0;i<R;++i) for(int j=0;j<C;++j){
//         float z=X->value(i,j);
//         g(i,j)= gy(i,j) * (z>0.f ? 1.f : a);
//     }
//     X->grad.add_( g );
// }

// // ----- matmul -----
// void vjp_MatMul(Node* n, const Tensor& gy){
//     Node* A = n->inputs[0].get();
//     Node* B = n->inputs[1].get();
//     const Tensor& At = A->value;
//     const Tensor& Bt = B->value;
//     auto [M, K]  = At.shape();
//     auto [K2, N] = Bt.shape();
//     (void)K2;

//     // MODIFIED: Add device dispatch
//     if (A->value.is_cpu()) {
//         // --- Existing CPU Logic ---
//         auto* mm = ag::kernels::cpu().matmul;
//         if (A->requires_grad){
//             Tensor BT = Tensor::transpose(Bt);
//             Tensor dA(M, K);
//             if (mm) { mm(gy.data(), BT.data(), dA.data(), M, N, K); }
//             else { dA = Tensor::matmul(gy, BT); }
//             A->grad.add_(dA);
//         }
//         if (B->requires_grad){
//             Tensor AT = Tensor::transpose(At);
//             Tensor dB(K, N);
//             if (mm) { mm(AT.data(), gy.data(), dB.data(), K, M, N); }
//             else { dB = Tensor::matmul(AT, gy); }
//             B->grad.add_(dB);
//         }
//     } else {
//         // --- NEW: CUDA Logic ---
//         // The CUDA kernel will overwrite gA and gB. This is correct if grads
//         // are zero-initialized before backward(), but a true accumulating
//         // version would require another buffer and an add kernel.
//         // For now, this is a correct and performant first step.
//         ag::kernels::cuda().vjp_matmul(
//             A->grad.data(), B->grad.data(), gy.data(),
//             At.data(), Bt.data(),
//             M, K, N, ag::current_stream()
//         );
//     }
// }

// void vjp_Dyntanh(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get(); 
//     Node* A = n->inputs[1].get(); 
//     Node* B = n->inputs[2].get(); 
//     Node* G = n->inputs[3].get();



//         if (X->requires_grad) X->grad.add_(gy*Tensor::sech(X->value*A->value)*Tensor::sech(X->value*A->value)*A->value*G->value); 
//     if (A->requires_grad) A->grad.add_(gy*Tensor::sech(X->value*A->value)*Tensor::sech(X->value*A->value)*X->value*G->value);
//     if (B->requires_grad) B->grad.add_(gy);
//     if (G->requires_grad) G->grad.add_(gy*Tensor::tanh(*(n->tape.back()))   );
// }

// // ----- reductions -----
// void vjp_Sum(Node* n, const Tensor& gy){
//     Node* X=n->inputs[0].get(); if(!X->requires_grad) return;
//     float s = gy(0,0);
//     X->grad.add_( Tensor::ones_like(X->value) * s );
// }
// void vjp_RowSum(Node* n, const Tensor& gy){
//     Node* X=n->inputs[0].get(); if(!X->requires_grad) return;
//     // gy [B,1] broadcast across columns
//     Tensor g = gy * Tensor::ones_like(X->value);
//     X->grad.add_( g );
// }
// void vjp_RowMax(Node* n, const Tensor& gy){
//     Node* X=n->inputs[0].get(); if(!X->requires_grad) return;
//     int R=X->value.rows(), C=X->value.cols();
//     Tensor m = Tensor::row_max(X->value);
//     Tensor g(R,C);
//     for(int i=0;i<R;++i) for(int j=0;j<C;++j)
//         g(i,j) = (X->value(i,j)==m(i,0)) ? gy(i,0) : 0.f;
//     X->grad.add_( g );
// }
// void vjp_MeanAll(Node* n, const Tensor& gy){
//     Node* X=n->inputs[0].get(); if(!X->requires_grad) return;
//     float scale = gy(0,0) / float(X->value.rows()*X->value.cols());
//     X->grad.add_( Tensor::ones_like(X->value) * scale );
// }

// // ----- softmax / losses -----
// void vjp_SoftmaxRow(Node* n, const Tensor& gy){
//     Node* Z = n->inputs[0].get(); if(!Z->requires_grad) return;
//     Tensor y = n->value; // softmax(Z)
//     Tensor dot = Tensor::row_sum( y * gy ); // [B,1]
//     Tensor g = y * (gy - dot);
//     Z->grad.add_( g );
// }
// void vjp_LogSumExpRow(Node* n, const Tensor& gy){
//     Node* Z = n->inputs[0].get(); if(!Z->requires_grad) return;
//     Tensor y = Tensor::softmax_row(Z->value);
//     Z->grad.add_( y * gy ); // gy [B,1] broadcast
// }
// void vjp_CeWithLogits(Node* n, const Tensor& gy /*unused: scalar gy*/){
//     Node* Z = n->inputs[0].get();
//     Node* Y = n->inputs[1].get();
//     int B = Z->value.rows();
//     Tensor sm = Tensor::softmax_row(Z->value);
//     Tensor gZ = (sm - Y->value) * (1.0f / float(B));
//     if (Z->requires_grad) Z->grad.add_( gZ );
//     if (Y->requires_grad) {
//         Tensor lse = Tensor::logsumexp_row(Z->value);
//         Tensor lsm = Z->value - lse;
//         Tensor gY  = lsm * (-1.0f / float(B));
//         Y->grad.add_( gY );
//     }
// }



// void vjp_KLDivergence(Node* n, const Tensor& gy /*unused: scalar gy*/){
//     Node* Z = n->inputs[0].get();
//     Node* Y = n->inputs[1].get();
//     int B = Z->value.rows();
//     Tensor sm = Tensor::softmax_row(Z->value);
//     Tensor gZ = (sm - Y->value) * (1.0f / float(B));
//     if (Z->requires_grad) Z->grad.add_( gZ );
//     if (Y->requires_grad) {
//         Tensor lse = Tensor::logsumexp_row(Z->value);
//         Tensor lsm = Z->value - lse;
//         Tensor gY = (Tensor::log(Y->value) + Tensor::ones_like(Y->value) - lsm) * (1.0f / float(B));
//         Y->grad.add_( gY );
//     }
// }

// void vjp_Div(Node* n, const Tensor& gy){
//     Node* A = n->inputs[0].get(); Node* B = n->inputs[1].get();
//     if (A->requires_grad) A->grad.add_( rt( gy * (Tensor::reciprocal(B->value)), A->value) );
//     if (B->requires_grad) B->grad.add_( rt( -gy * (Tensor::reciprocal(B->value)) * (Tensor::reciprocal(B->value)) * A->value, B->value) );
// }
// void vjp_Reciprocal(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get(); if (!X->requires_grad) return;
//     X->grad.add_( rt( -gy * (Tensor::reciprocal(X->value)) * (Tensor::reciprocal(X->value)), X->value) );
// }





// void vjp_Linear(Node* n, const Tensor& gy){
//     Node* A = n->inputs[0].get();
//     Node* B = n->inputs[1].get();
//     Node* C = n->inputs[2].get();

//     // External kernel (if plugin loaded), else fallback to Tensor::matmul
//     auto* mm = ag::kernels::cpu().matmul;

//     // Shapes
//     const Tensor& At = A->value;
//     const Tensor& Bt = B->value;
//     auto [M, K]  = At.shape();
//     auto [K2, N] = Bt.shape();
//     (void)K2; // assume forward already checked

//     if (A->requires_grad){
//         Tensor BT = Bt; // (N x K)
//         Tensor dA(M, K);                   // temp buffer

//         if (mm) {
//             // dA = gy (MxN) * BT (NxK)
//             mm(gy.data(), BT.data(), dA.data(), M, N, K);
//         } else {
//             dA = Tensor::matmul(gy, BT);
//         }
//         A->grad.add_(dA);
//     }

//     if (B->requires_grad){
//         Tensor AT = Tensor::transpose(At); // (K x M)
//         Tensor dB(K, N);                   // temp buffer

//         if (mm) {
//             // dB = AT (KxM) * gy (MxN)
//             mm(AT.data(), gy.data(), dB.data(), K, M, N);
//         } else {
//             dB = Tensor::matmul(AT, gy);
//         }
//         B->grad.add_(dB);
//     }
//     if (C->requires_grad) C->grad.add_( rt(gy, C->value) );
// }






// void vjp_Cosh(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (X->requires_grad) X->grad.add_( rt( gy * (Tensor::sinh(X->value)), X->value) );
// }


// void vjp_Sinh(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (X->requires_grad) X->grad.add_( rt( gy * (Tensor::cosh(X->value)), X->value) );
// }


// void vjp_Sign(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (!X->requires_grad) return;
//     Tensor th = n->value, one = Tensor::ones_like(th);
    
//     X->grad.add_( rt( gy * 0.0f, X->value) );
// }

// void vjp_Cos(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (X->requires_grad) X->grad.add_( rt( -1.0 * gy* (Tensor::sin(X->value)), X->value) );
// }


// void vjp_Sin(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (X->requires_grad) X->grad.add_( rt( gy * (Tensor::cos(X->value)), X->value) );
// }


// void vjp_Sqrt(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (X->requires_grad) X->grad.add_( rt(0.5f * gy * Tensor::sqrt(  Tensor::reciprocal(X->value)), X->value) );
// }

// void vjp_Relumask(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (!X->requires_grad) return;
//     Tensor th = n->value, one = Tensor::ones_like(th);

//     X->grad.add_( rt( gy * 0.0f, X->value) );
// }















// void vjp_RELUAtt(Node* n, const Tensor& gy){
//     Node* A = n->inputs[0].get();
//     Node* B = n->inputs[1].get();
//     Node* C = n->inputs[2].get();
//     Node* D = n->inputs[3].get();

//     Tensor q = n->tape[0] ? *n->tape[0] : Tensor();
//     Tensor k = n->tape[1] ? *n->tape[1] : Tensor();
//     Tensor v = n->tape[2] ? *n->tape[2] : Tensor();
//     float scale = 1.0f / std::sqrt(float(k.cols()));
//     Tensor s = n->tape[3] ? *n->tape[3] : Tensor();

//     // ---- Backprop chain ----

//     // y = s v
//     Tensor dL_ds = Tensor::matmul(gy, Tensor::transpose(v));   // [B x B]
//     Tensor dL_dv = Tensor::matmul(Tensor::transpose(s), gy);   // [A x D]

//     // s = softmax(g)
//     Tensor dL_dg; 
//     {
//         Tensor dot = Tensor::relu_mask(s )* dL_ds;
//         dL_dg = dot;
//     }

//     // g = q k^T
//     Tensor dL_dq = Tensor::matmul(dL_dg, k);
//     Tensor dL_dk = Tensor::matmul(Tensor::transpose(dL_dg), q);

// // q = A B^T
// Tensor dL_dA_q = Tensor::matmul(dL_dq, B->value) * scale;
// Tensor dL_dB   = Tensor::matmul(Tensor::transpose(dL_dq), A->value) * scale;

// // k = A C^T
// Tensor dL_dA_k = Tensor::matmul(dL_dk, C->value) * scale;
// Tensor dL_dC   = Tensor::matmul(Tensor::transpose(dL_dk), A->value) * scale;

// // v = A D^T
// Tensor dL_dA_v = Tensor::matmul(dL_dv, D->value);
// Tensor dL_dD   = Tensor::matmul(Tensor::transpose(dL_dv), A->value);

//     // combine A contributions
//     Tensor dL_dA = dL_dA_q + dL_dA_k + dL_dA_v;

//     // ---- Accumulate ----
//     if (A->requires_grad) A->grad.add_(dL_dA);
//     if (B->requires_grad) B->grad.add_(dL_dB);
//     if (C->requires_grad) C->grad.add_(dL_dC);
//     if (D->requires_grad) D->grad.add_(dL_dD);


// }






// void vjp_MOE(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     Node* W = n->inputs[1].get();
//     Node* B = n->inputs[2].get();

//     Tensor y = Tensor::matmul(X->value, Tensor::transpose(W->value)) + B->value; 

//     Tensor dL_dB = gy;
//     Tensor dL_dW = Tensor::matmul(Tensor::transpose(gy), X->value);
//     Tensor dL_dX = Tensor::matmul(gy, W->value);

//     if (X->requires_grad) X->grad.add_(dL_dX);
//     if (W->requires_grad) W->grad.add_(dL_dW);
//     if (B->requires_grad) B->grad.add_(dL_dB);

// }



// void vjp_SigAtt(Node* n, const Tensor& gy){
//     Node* A = n->inputs[0].get();
//     Node* B = n->inputs[1].get();
//     Node* C = n->inputs[2].get();
//     Node* D = n->inputs[3].get();

//     Tensor q = n->tape[0] ? *n->tape[0] : Tensor();
//     Tensor k = n->tape[1] ? *n->tape[1] : Tensor();
//     Tensor v = n->tape[2] ? *n->tape[2] : Tensor();
//     float scale = 1.0f / std::sqrt(float(k.cols()));
//     Tensor s = n->tape[3] ? *n->tape[3] : Tensor();

//     // ---- Backprop chain ----

//     // y = s v
//     Tensor dL_ds = Tensor::matmul(gy, Tensor::transpose(v));   // [B x B]
//     Tensor dL_dv = Tensor::matmul(Tensor::transpose(s), gy);   // [A x D]

//     // s = softmax(g)
//     Tensor dL_dg; 
//     {
//         Tensor dot = ( s * (Tensor::ones_like(s)-s))* dL_ds;

//         dL_dg = dot;
//     }

//     // g = q k^T
//     Tensor dL_dq = Tensor::matmul(dL_dg, k);
//     Tensor dL_dk = Tensor::matmul(Tensor::transpose(dL_dg), q);

// // q = A B^T
// Tensor dL_dA_q = Tensor::matmul(dL_dq, B->value) * scale;
// Tensor dL_dB   = Tensor::matmul(Tensor::transpose(dL_dq), A->value) * scale;

// // k = A C^T
// Tensor dL_dA_k = Tensor::matmul(dL_dk, C->value) * scale;
// Tensor dL_dC   = Tensor::matmul(Tensor::transpose(dL_dk), A->value) * scale;

// // v = A D^T
// Tensor dL_dA_v = Tensor::matmul(dL_dv, D->value);
// Tensor dL_dD   = Tensor::matmul(Tensor::transpose(dL_dv), A->value);

//     // combine A contributions
//     Tensor dL_dA = dL_dA_q + dL_dA_k + dL_dA_v;

//     // ---- Accumulate ----
//     if (A->requires_grad) A->grad.add_(dL_dA);
//     if (B->requires_grad) B->grad.add_(dL_dB);
//     if (C->requires_grad) C->grad.add_(dL_dC);
//     if (D->requires_grad) D->grad.add_(dL_dD);


// }


// void vjp_MSELoss(Node* n, const Tensor& gy /*unused: scalar gy*/){
//     Node* Z = n->inputs[0].get();
//     Node* Y = n->inputs[1].get();
//     int B = Z->value.rows(), C = Z->value.cols();
//     Tensor diff = Z->value - Y->value;
//     Tensor gZ = diff * (2.0f / float(B * C));
//     Tensor gY = -diff * (2.0f / float(B * C));
//     if (Z->requires_grad) Z->grad.add_(gZ);
//     if (Y->requires_grad) Y->grad.add_(gY);
// }

// void vjp_MAELoss(Node* n, const Tensor& gy /*unused: scalar gy*/){
//     Node* Z = n->inputs[0].get();
//     Node* Y = n->inputs[1].get();
//     int B = Z->value.rows(), C = Z->value.cols();
//     Tensor diff = Tensor::sign(Z->value - Y->value);
//     Tensor gZ = diff * (1.0f / float(B * C));
//     Tensor gY = -diff * (1.0f / float(B * C));
//     if (Z->requires_grad) Z->grad.add_(gZ);
//     if (Y->requires_grad) Y->grad.add_(gY);
// }


// void vjp_Leaf(Node*, const Tensor&){ /* no-op */ }


// } // anon


// // -------- dispatch table --------
// VjpFn vjp_lookup(Op op){
//     switch(op){
// #define OP(name, arity, str) case Op::name: return &detail::vjp_##name;
// #include "ad/detail/ops.def"
// #undef OP
//         default: return nullptr;
//     }
// }

// } // namespace ag
// ====================================================================
// FILE: cgadimpl/src/autodiff/autodiff_vjp_ops.cpp (GPU-Aware Version)
// ====================================================================

#include "ad/detail/autodiff_ops.hpp"
#include "ad/runtime.hpp"
#include <cmath>
#include <stdexcept> // Required for std::runtime_error

namespace ag {
namespace detail{

// helper: reduce a gradient to a parent's shape (broadcast-aware)
inline Tensor rt(const Tensor& g, const Tensor& like){ return Tensor::reduce_to(g, like); }

// ----- elementwise binary -----
void vjp_Add(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();

    if (A->value.is_cpu()) {
        if (A->requires_grad) A->grad.add_( rt(gy, A->value) );
        if (B->requires_grad) B->grad.add_( rt(gy, B->value) );
    } else {
        ag::kernels::cuda().vjp_add(A->grad.data(), B->grad.data(), gy.data(),
                                    gy.numel(), ag::current_stream());
    }
}
void vjp_Sub(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get(); Node* B = n->inputs[1].get();
    if (n->value.is_cpu()) {
        if (A->requires_grad) A->grad.add_( rt(gy, A->value) );
        if (B->requires_grad) B->grad.add_( rt(-gy, B->value) );
    } else {
        throw std::runtime_error("VJP for Sub on CUDA not implemented yet!");
    }
}
void vjp_Mul(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get(); Node* B = n->inputs[1].get();
    if (n->value.is_cpu()) {
        if (A->requires_grad) A->grad.add_( rt( gy * B->value, A->value) );
        if (B->requires_grad) B->grad.add_( rt( gy * A->value, B->value) );
    } else {
        throw std::runtime_error("VJP for Mul on CUDA not implemented yet!");
    }
}

// ----- elementwise trinary & matmul -----
void vjp_FMA(Node* n, const Tensor& gy){
    if (n->value.is_cpu()) {
        Node* A = n->inputs[0].get();
        Node* B = n->inputs[1].get();
        Node* C = n->inputs[2].get();
        
        const Tensor& At = A->value;
        const Tensor& Bt = B->value;

        if (A->requires_grad){
            A->grad.add_(Tensor::matmul(gy, Tensor::transpose(Bt)));
        }
        if (B->requires_grad){
            B->grad.add_(Tensor::matmul(Tensor::transpose(At), gy));
        }
        if (C->requires_grad) C->grad.add_( rt(gy, C->value) );
    } else {
        throw std::runtime_error("VJP for FMA on CUDA not implemented yet!");
    }
}

// ----- Normalization Layers -----
void vjp_LayerNorm(Node* n, const Tensor& gy){
    if (n->value.is_cpu()) {
        Node* x = n->inputs[0].get();
        int N = x->value.cols();
        Tensor std_dev = Tensor::sqrt(*(n->tape[0]) + 0.01);
        Tensor xmu = x->value - *(n->tape[1]);
        Tensor grad_sum = Tensor::row_sum(gy);
        Tensor grad_dot_xmu = Tensor::row_sum(gy * xmu);
        Tensor term1 = gy * float(N);
        Tensor term2 = term1 - grad_sum;
        Tensor term3 = term2 - (xmu * (grad_dot_xmu / (*(n->tape[0]) + 0.01)));
        Tensor dx = term3 / (std_dev * float(N));
        if (x->requires_grad) x->grad.add_(dx);
    } else {
        throw std::runtime_error("VJP for LayerNorm on CUDA not implemented yet!");
    }
}

void vjp_RealLayerNorm(Node* n, const Tensor& gy){
    if (n->value.is_cpu()) {
        Node* x = n->inputs[0].get();
        Node* b = n->inputs[1].get();
        Node* g = n->inputs[2].get();
        int N = x->value.cols();
        Tensor std_dev = Tensor::sqrt(*(n->tape[0]) + 0.01);
        Tensor xmu = x->value - *(n->tape[1]);
        Tensor grad_sum = Tensor::row_sum(gy);
        Tensor grad_dot_xmu = Tensor::row_sum(gy * xmu);
        Tensor term1 = gy * float(N);
        Tensor term2 = term1 - grad_sum;
        Tensor term3 = term2 - (xmu * (grad_dot_xmu / (*(n->tape[0]) + 0.01)));
        Tensor dx = term3 / (std_dev * float(N));
        if (x->requires_grad) x->grad.add_(dx);
        if (b->requires_grad) b->grad.add_(Tensor::row_sum(gy));
        if (g->requires_grad) g->grad.add_(Tensor::row_sum(gy * (*(n->tape[2]))));
    } else {
        throw std::runtime_error("VJP for RealLayerNorm on CUDA not implemented yet!");
    }
}

void vjp_RMSNorm(Node* n, const Tensor& gy){
    if (n->value.is_cpu()) {
        Node* x = n->inputs[0].get();
        Tensor rms = *n->tape[0];
        Tensor y   = *n->tape[1];
        Tensor dot = Tensor::row_sum(gy * y);
        Tensor grad_x = (gy / rms) - (y * dot / (rms*x->value.cols()));
        if (x->requires_grad) x->grad.add_(grad_x);
    } else {
        throw std::runtime_error("VJP for RMSNorm on CUDA not implemented yet!");
    }
}

void vjp_RealRMSNorm(Node* n, const Tensor& gy){
    if (n->value.is_cpu()) {
        Node* x = n->inputs[0].get();
        Node* g = n->inputs[1].get();
        Tensor rms = *n->tape[0];
        Tensor y   = *n->tape[1];
        Tensor dot = Tensor::row_sum(gy * y);
        Tensor grad_x = g->value * ((gy / rms) - (y * dot / (rms*x->value.cols())));
        if (x->requires_grad) x->grad.add_(grad_x);
        if (g->requires_grad) g->grad.add_(gy * (x->value / rms));
    } else {
        throw std::runtime_error("VJP for RealRMSNorm on CUDA not implemented yet!");
    }
}

// ----- Attention Mechanisms -----
void vjp_Attention(Node* n, const Tensor& gy){
    if (n->value.is_cpu()) {
        Node* A = n->inputs[0].get();
        Node* B = n->inputs[1].get();
        Node* C = n->inputs[2].get();
        Node* D = n->inputs[3].get();
        Tensor q = *n->tape[0], k = *n->tape[1], v = *n->tape[2], s = *n->tape[3];
        float scale = 1.0f / std::sqrt(float(k.cols()));
        Tensor dL_ds = Tensor::matmul(gy, Tensor::transpose(v));
        Tensor dL_dv = Tensor::matmul(Tensor::transpose(s), gy);
        Tensor dot = Tensor::row_sum(s * dL_ds);
        Tensor dL_dg = s * (dL_ds - dot);
        Tensor dL_dq = Tensor::matmul(dL_dg, k);
        Tensor dL_dk = Tensor::matmul(Tensor::transpose(dL_dg), q);
        Tensor dL_dA_q = Tensor::matmul(dL_dq, Tensor::transpose(B->value));
        Tensor dL_dB   = Tensor::matmul(Tensor::transpose(A->value), dL_dq) * scale;
        Tensor dL_dA_k = Tensor::matmul(dL_dk, Tensor::transpose(C->value));
        Tensor dL_dC   = Tensor::matmul(Tensor::transpose(A->value), dL_dk) * scale;
        Tensor dL_dA_v = Tensor::matmul(dL_dv, Tensor::transpose(D->value));
        Tensor dL_dD   = Tensor::matmul(Tensor::transpose(A->value), dL_dv);
        if (A->requires_grad) A->grad.add_(dL_dA_q + dL_dA_k + dL_dA_v);
        if (B->requires_grad) B->grad.add_(dL_dB);
        if (C->requires_grad) C->grad.add_(dL_dC);
        if (D->requires_grad) D->grad.add_(dL_dD);
    } else {
        throw std::runtime_error("VJP for Attention on CUDA not implemented yet!");
    }
}

void vjp_AlibiAttention(Node* n, const Tensor& gy){
    // Assuming same VJP as Attention for now
    vjp_Attention(n, gy);
}

void vjp_SWIGLU(Node* n, const Tensor& gy){
    if (n->value.is_cpu()) {
        Node* X = n->inputs[0].get();
        Node* A = n->inputs[1].get();
        Node* B = n->inputs[2].get();
        Node* C = n->inputs[3].get();
        Node* D = n->inputs[4].get();
        Tensor y = Tensor::matmul(X->value, Tensor::transpose(A->value)) + B->value;
        Tensor q = y * Tensor::sigmoid(y);
        Tensor h = Tensor::matmul(X->value, Tensor::transpose(C->value)) + D->value;
        Tensor Swishdif = Tensor::sigmoid(y) + y * (Tensor::sigmoid(y) * (Tensor::ones_like(y) - Tensor::sigmoid(y)));
        Tensor dL_dB = Swishdif * h * gy;
        Tensor dL_dA = Tensor::matmul(Tensor::transpose(dL_dB), X->value);
        Tensor dL_dD = q * gy;
        Tensor dL_dC = Tensor::matmul(Tensor::transpose(dL_dD), X->value);
        Tensor dL_dX = Tensor::matmul(dL_dB, A->value) + Tensor::matmul(dL_dD, C->value);
        if (X->requires_grad) X->grad.add_(dL_dX);
        if (A->requires_grad) A->grad.add_(dL_dA);
        if (B->requires_grad) B->grad.add_(dL_dB);
        if (C->requires_grad) C->grad.add_(dL_dC);
        if (D->requires_grad) D->grad.add_(dL_dD);
    } else {
        throw std::runtime_error("VJP for SWIGLU on CUDA not implemented yet!");
    }
}

// ----- Unary Activations -----

// void vjp_Relu(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (!X->requires_grad) return;

//     if (X->value.is_cpu()) {
//         Tensor dA = Tensor::relu_mask(X->value);
//         X->grad.add_( rt( gy * dA, X->value) );
//     } else {
//         // ===================== THIS IS THE CHANGE =====================
//         // REPLACE THIS:
//         // throw std::runtime_error("VJP for Relu on CUDA not implemented yet!");
        
//         // WITH THIS:
//         ag::kernels::cuda().vjp_relu(
//             X->grad.data(),      // gX (output)
//             gy.data(),           // gy (input)
//             X->value.data(),     // X (original input)
//             gy.numel(),          // n
//             ag::current_stream() // s
//         );
//         // =============================================================
//     }
// }
void vjp_Relu(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;

    if (X->value.is_cpu()) {
        auto fn = ag::kernels::cpu().relu_bwd;
        if (fn) {
            // --- NEW: Call the fast backward kernel ---
            // The kernel overwrites the output, so we compute into a temporary
            // buffer and then add it to the gradient to ensure accumulation.
            Tensor dX_temp = Tensor::zeros_like(X->value);
            fn(X->value.data(), gy.data(), dX_temp.data(), X->value.numel());
            X->grad.add_(dX_temp);
        } else {
            // --- OLD: Fallback to generic C++ ---
            Tensor dA = Tensor::relu_mask(X->value);
            X->grad.add_( rt( gy * dA, X->value) );
        }
    } else {
        // GPU path (when ready)
        throw std::runtime_error("ReLU backward on CUDA not implemented yet!");
    }
}

// void vjp_Exp(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (!X->requires_grad) return;
//     if (n->value.is_cpu()) {
//         X->grad.add_( rt( gy * n->value, X->value) );
//     } else {
//         throw std::runtime_error("VJP for Exp on CUDA not implemented yet!");
//     }
// }


    void vjp_Exp(Node* n, const Tensor& gy){
        Node* X = n->inputs[0].get();
        if (!X->requires_grad) return;

        if (X->value.is_cpu()) {
            auto fn = ag::kernels::cpu().exp_bwd_from_y;
            if (fn) {
                // --- NEW: Call the fast backward kernel ---
                // The kernel is the efficient `_from_y` version, which takes the output
                // of the forward pass (`y = exp(x)`). This is stored in `n->value`.
                
                Tensor dX_temp = Tensor::zeros_like(X->value);
                fn(n->value.data(), gy.data(), dX_temp.data(), X->value.numel());
                X->grad.add_(dX_temp); // Accumulate gradient
            } else {
                // --- OLD: Fallback to generic C++ ---
                // The derivative is just the output of the forward pass (n->value)
                X->grad.add_( rt( gy * n->value, X->value) );
            }
        } else {
            // GPU path (when ready)
            throw std::runtime_error("Exp backward on CUDA not implemented yet!");
        }
    }

    // void vjp_Log(Node* n, const Tensor& gy){
    //     Node* X = n->inputs[0].get();
    //     if (!X->requires_grad) return;
    //     if (n->value.is_cpu()) {
    //         if (X->requires_grad) X->grad.add_( rt( gy / X->value, X->value) );
    //     } else {
    //         throw std::runtime_error("VJP for Log on CUDA not implemented yet!");
    //     }
    // }


void vjp_Log(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;

    if (X->value.is_cpu()) {
        auto fn = ag::kernels::cpu().log_bwd;
        if (fn) {
            // --- NEW: Call the fast backward kernel ---
            // The kernel takes the original input `x`.
            Tensor dX_temp = Tensor::zeros_like(X->value);
            fn(X->value.data(), gy.data(), dX_temp.data(), X->value.numel());
            X->grad.add_(dX_temp); // Accumulate gradient
        } else {
            // --- OLD: Fallback to generic C++ ---
            X->grad.add_( rt( gy / X->value, X->value) );
        }
    } else {
        // GPU path (when ready)
        throw std::runtime_error("Log backward on CUDA not implemented yet!");
    }
}


    void vjp_GCU(Node* n, const Tensor& gy){
        Node* X = n->inputs[0].get();
        if (!X->requires_grad) return;
        if (n->value.is_cpu()) {
            X->grad.add_( rt( gy * (Tensor::cos(X->value)-(X->value*Tensor::sin(X->value))), X->value) );
        } else {
            throw std::runtime_error("VJP for GCU on CUDA not implemented yet!");
        }
    }

    void vjp_Mish(Node* n, const Tensor& gy){
        Node* X = n->inputs[0].get();
        if (!X->requires_grad) return;
        if (n->value.is_cpu()) {
            Tensor sp = Tensor::softplus(X->value);
            Tensor th = Tensor::tanh(sp);
            Tensor sig = Tensor::sigmoid(X->value);
            X->grad.add_( rt( gy * (th + (X->value * sig * (Tensor::ones_like(th) - th*th))), X->value) );
        } else {
            throw std::runtime_error("VJP for Mish on CUDA not implemented yet!");
        }
    }

// void vjp_Tanh(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (!X->requires_grad) return;
//     if (n->value.is_cpu()) {
//         Tensor th = n->value;
//         X->grad.add_( rt( gy * (Tensor::ones_like(th) - th*th), X->value) );
//     } else {
//         throw std::runtime_error("VJP for Tanh on CUDA not implemented yet!");
//     }
// }

void vjp_Tanh(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;

    if (X->value.is_cpu()) {
        auto fn = ag::kernels::cpu().tanh_bwd_from_t;
        if (fn) {
            // --- NEW: Call the fast backward kernel ---
            // The kernel is the efficient `_from_t` version, which takes the output
            // of the forward pass (`t = tanh(x)`) as input. This is stored in `n->value`.
            
            Tensor dX_temp = Tensor::zeros_like(X->value);
            fn(n->value.data(), gy.data(), dX_temp.data(), X->value.numel());
            X->grad.add_(dX_temp); // Accumulate gradient
        } else {
            // --- OLD: Fallback to generic C++ ---
            Tensor th = n->value;
            Tensor one = Tensor::ones_like(th);
            X->grad.add_( rt( gy * (one - th*th), X->value) );
        }
    } else {
        // GPU path (when ready)
        throw std::runtime_error("Tanh backward on CUDA not implemented yet!");
    }
}

void vjp_Sigmoid(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;

    if (X->value.is_cpu()) {
        auto fn = ag::kernels::cpu().sigmoid_bwd_from_s;
        if (fn) {
            // --- NEW: Call the fast backward kernel ---
            // We use the more efficient kernel that takes `s` (the output of the
            // forward pass), which is stored in `n->value`.
            
            Tensor dX_temp = Tensor::zeros_like(X->value);
            fn(n->value.data(), gy.data(), dX_temp.data(), X->value.numel());
            X->grad.add_(dX_temp); // Accumulate gradient
        } else {
            // --- OLD: Fallback to generic C++ ---
            const Tensor& Y = n->value; // This is 's'
            Tensor dA = Y * (Tensor::ones_like(Y) - Y);
            X->grad.add_( rt( gy * dA, X->value) );
        }
    } else {
        // GPU path (when ready)
        throw std::runtime_error("Sigmoid backward on CUDA not implemented");
    }
}

// void vjp_Softplus(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (!X->requires_grad) return;
//     if (n->value.is_cpu()) {
//         X->grad.add_( rt( gy * Tensor::sigmoid(X->value), X->value) );
//     } else {
//         throw std::runtime_error("VJP for Softplus on CUDA not implemented yet!");
//     }
// }

void vjp_Softplus(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;

    if (X->value.is_cpu()) {
        auto fn = ag::kernels::cpu().softplus_bwd;
        if (fn) {
            // --- NEW: Call the fast backward kernel ---
            // The kernel takes the original input `x` to compute the derivative (sigmoid(x)).
            Tensor dX_temp = Tensor::zeros_like(X->value);
            fn(X->value.data(), gy.data(), dX_temp.data(), X->value.numel());
            X->grad.add_(dX_temp); // Accumulate gradient
        } else {
            // --- OLD: Fallback to generic C++ ---
            X->grad.add_( rt( gy * Tensor::sigmoid(X->value), X->value) );
        }
    } else {
        // GPU path (when ready)
        throw std::runtime_error("Softplus backward on CUDA not implemented yet!");
    }
}

void vjp_Gaus(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;
    if (n->value.is_cpu()) {
        X->grad.add_( rt( gy * -2.0f * X->value * Tensor::exp(-1.0f * X->value * X->value), X->value) );
    } else {
        throw std::runtime_error("VJP for Gaus on CUDA not implemented yet!");
    }
}

void vjp_Transpose(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;
    if (n->value.is_cpu()) {
        X->grad.add_( rt( Tensor::transpose(gy) , X->value) );
    } else {
        throw std::runtime_error("VJP for Transpose on CUDA not implemented yet!");
    }
}

void vjp_SiLU(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;
    if (n->value.is_cpu()) {
        Tensor s = Tensor::sigmoid(X->value);
        X->grad.add_( rt( gy * ( s + X->value * ( s * (Tensor::ones_like(s)-s) ) ), X->value) );
    } else {
        throw std::runtime_error("VJP for SiLU on CUDA not implemented yet!");
    }
}

void vjp_Parcon(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;
    if (n->value.is_cpu()) {
        X->grad.add_( rt( gy * ( 2.0f * Tensor::ones_like(X->value) - 2.0f * X->value  ), X->value) );
    } else {
        throw std::runtime_error("VJP for Parcon on CUDA not implemented yet!");
    }
}

void vjp_LiSHT(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;
    if (n->value.is_cpu()) {
        Tensor sech_x = Tensor::sech(X->value);
        X->grad.add_( rt( gy * ( Tensor::tanh(X->value) + (sech_x * sech_x * X->value ) ), X->value) );
    } else {
        throw std::runtime_error("VJP for LiSHT on CUDA not implemented yet!");
    }
}

// void vjp_GELU(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (!X->requires_grad) return;
//     if (n->value.is_cpu()) {
//         constexpr float c = 0.79788456080286535588f; // sqrt(2/pi)
//         Tensor x=X->value;
//         Tensor u = c * (x + 0.044715f * x*x*x);
//         Tensor dudx = c * (1.f + 0.134145f * x*x);
//         Tensor th=Tensor::tanh(u);
//         Tensor one=Tensor::ones_like(th);
//         Tensor dgelu=(one+th)*0.5f + (x * ((one - th*th) * dudx))*0.5f;
//         X->grad.add_( rt( gy * dgelu, X->value) );
//     } else {
//         throw std::runtime_error("VJP for GELU on CUDA not implemented yet!");
//     }
// }
void vjp_GELU(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;

    if (X->value.is_cpu()) {
        auto fn = ag::kernels::cpu().gelu_bwd;
        if (fn) {
            // --- NEW: Call the fast backward kernel ---
            fn(X->value.data(), gy.data(), X->grad.data(), X->value.numel());
        } else {
            constexpr float c = 0.79788456080286535588f; // sqrt(2/pi)
            Tensor x=X->value;
            Tensor u = c * (x + 0.044715f * x*x*x);
            Tensor dudx = c * (1.f + 0.134145f * x*x);
            Tensor th=Tensor::tanh(u);
            Tensor one=Tensor::ones_like(th);
            Tensor dgelu=(one+th)*0.5f + (x * ((one - th*th) * dudx))*0.5f;
            X->grad.add_( rt( gy * dgelu, X->value) );
        }
    } else {
        // GPU path (when ready)
        throw std::runtime_error("GELU backward on CUDA not implemented");
    }
}

void vjp_LeakyRelu(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;

    // Get alpha from the second input node in the graph. This is correct.
    Node* A_node = n->inputs[1].get();
    float alpha = A_node->value(0,0);

    if (X->value.is_cpu()) {
        auto fn = ag::kernels::cpu().leakyrelu_bwd;
        if (fn) {
            // --- NEW: Call the fast backward kernel ---
            // IMPORTANT: We compute into a temporary buffer first, then add.
            // This correctly handles gradient accumulation.
            Tensor dX_temp = Tensor::zeros_like(X->value);
            fn(X->value.data(), gy.data(), dX_temp.data(), X->value.numel(), alpha);
            X->grad.add_(dX_temp);
        } else {
            // --- OLD: Fallback to the slow loop ---
            int R=X->value.rows(), C=X->value.cols();
            Tensor g(R,C);
            for(int i=0;i<R;++i) for(int j=0;j<C;++j){
                float z=X->value(i,j);
                g(i,j) = gy(i,j) * (z>0.f ? 1.f : alpha);
            }
            X->grad.add_( g );
        }
    } else {
        // GPU path (when ready)
        throw std::runtime_error("LeakyReLU backward on CUDA not implemented");
    }
}

// ----- MatMul -----
void vjp_MatMul(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get();
    Node* B = n->inputs[1].get();
    const Tensor& At = A->value;
    const Tensor& Bt = B->value;

    if (At.is_cpu()) {
        if (A->requires_grad){
            A->grad.add_(Tensor::matmul(gy, Tensor::transpose(Bt)));
        }
        if (B->requires_grad){
            B->grad.add_(Tensor::matmul(Tensor::transpose(At), gy));
        }
    } else {
        auto [M, K]  = At.shape();
        auto [K2, N] = Bt.shape();
        (void)K2;
        ag::kernels::cuda().vjp_matmul(
            A->grad.data(), B->grad.data(), gy.data(),
            At.data(), Bt.data(),
            M, K, N, ag::current_stream()
        );
    }
}

// void vjp_MatMul(Node* n, const Tensor& gy){
//     Node* A_node = n->inputs[0].get();
//     Node* B_node = n->inputs[1].get();
//     const Tensor& A = A_node->value;
//     const Tensor& B = B_node->value;
    
//     auto [M, K] = A.shape();
//     auto [K2, N] = B.shape();

//     if (A.is_cpu()) {
//         // --- NEW: Dispatch to optimized CPU backward kernels ---
//         auto fn_dA = ag::kernels::cpu().matmul_bwd_dA;
//         auto fn_dB = ag::kernels::cpu().matmul_bwd_dB;

//         if (A_node->requires_grad) {
//             Tensor dA_temp = Tensor::zeros_like(A);
//             if (fn_dA) {
//                 // gA(M,K) = gy(M,N) @ B^T(N,K)
//                 fn_dA(gy.data(), B.data(), dA_temp.data(), M, K, N);
//             } else {
//                 // Fallback
//                 dA_temp = Tensor::matmul(gy, Tensor::transpose(B));
//             }
//             A_node->grad.add_(dA_temp);
//         }

//         if (B_node->requires_grad) {
//             Tensor dB_temp = Tensor::zeros_like(B);
//             if (fn_dB) {
//                 // gB(K,N) = A^T(K,M) @ gy(M,N)
//                 fn_dB(A.data(), gy.data(), dB_temp.data(), M, K, N);
//             } else {
//                 // Fallback
//                 dB_temp = Tensor::matmul(Tensor::transpose(A), gy);
//             }
//             B_node->grad.add_(dB_temp);
//         }

//     } else { // GPU Path
//         // This part is already correct and uses your working CUDA VJP kernel.
//         auto fn = ag::kernels::cuda().vjp_matmul;
//         if (fn) {
//             fn(A_node->grad.data(), B_node->grad.data(), gy.data(),
//                A.data(), B.data(),
//                M, K, N, ag::current_stream());
//         } else {
//             throw std::runtime_error("MatMul backward on CUDA not implemented or loaded.");
//         }
//     }
// }


void vjp_Dyntanh(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get(); 
    Node* A = n->inputs[1].get(); 
    Node* B = n->inputs[2].get(); 
    Node* G = n->inputs[3].get();
    if (n->value.is_cpu()) {
        Tensor sech_val = Tensor::sech(X->value * A->value);
        if (X->requires_grad) X->grad.add_(gy * sech_val * sech_val * A->value * G->value); 
        if (A->requires_grad) A->grad.add_(gy * sech_val * sech_val * X->value * G->value);
        if (B->requires_grad) B->grad.add_(gy);
        if (G->requires_grad) G->grad.add_(gy * Tensor::tanh(*(n->tape.back())));
    } else {
        throw std::runtime_error("VJP for Dyntanh on CUDA not implemented yet!");
    }
}

// ----- Reductions -----
void vjp_Sum(Node* n, const Tensor& gy){
    Node* X=n->inputs[0].get();
    if (!X->requires_grad) return;
    if (n->value.is_cpu()) {
        float s = gy(0,0);
        X->grad.add_( Tensor::ones_like(X->value) * s );
    } else {
        throw std::runtime_error("VJP for Sum on CUDA not implemented yet!");
    }
}
void vjp_RowSum(Node* n, const Tensor& gy){
    Node* X=n->inputs[0].get();
    if (!X->requires_grad) return;
    if (n->value.is_cpu()) {
        X->grad.add_( gy * Tensor::ones_like(X->value) );
    } else {
        throw std::runtime_error("VJP for RowSum on CUDA not implemented yet!");
    }
}
void vjp_RowMax(Node* n, const Tensor& gy){
    Node* X=n->inputs[0].get();
    if (!X->requires_grad) return;
    if (n->value.is_cpu()) {
        Tensor m = Tensor::row_max(X->value);
        Tensor g = Tensor::zeros_like(X->value);
        for(int i=0; i<X->value.rows(); ++i) for(int j=0; j<X->value.cols(); ++j)
            g(i,j) = (X->value(i,j)==m(i,0)) ? gy(i,0) : 0.f;
        X->grad.add_( g );
    } else {
        throw std::runtime_error("VJP for RowMax on CUDA not implemented yet!");
    }
}
void vjp_MeanAll(Node* n, const Tensor& gy){
    Node* X=n->inputs[0].get();
    if (!X->requires_grad) return;
    if (n->value.is_cpu()) {
        float scale = gy(0,0) / float(X->value.numel());
        X->grad.add_( Tensor::ones_like(X->value) * scale );
    } else {
        throw std::runtime_error("VJP for MeanAll on CUDA not implemented yet!");
    }
}

// ----- Softmax / Losses -----
void vjp_SoftmaxRow(Node* n, const Tensor& gy){
    Node* Z = n->inputs[0].get();
    if (!Z->requires_grad) return;
    if (n->value.is_cpu()) {
        Tensor y = n->value;
        Tensor dot = Tensor::row_sum( y * gy );
        Z->grad.add_( y * (gy - dot) );
    } else {
        throw std::runtime_error("VJP for SoftmaxRow on CUDA not implemented yet!");
    }
}
void vjp_LogSumExpRow(Node* n, const Tensor& gy){
    Node* Z = n->inputs[0].get();
    if (!Z->requires_grad) return;
    if (n->value.is_cpu()) {
        Tensor y = Tensor::softmax_row(Z->value);
        Z->grad.add_( y * gy );
    } else {
        throw std::runtime_error("VJP for LogSumExpRow on CUDA not implemented yet!");
    }
}
void vjp_CeWithLogits(Node* n, const Tensor& gy){
    Node* Z = n->inputs[0].get();
    Node* Y = n->inputs[1].get();
    if (n->value.is_cpu()) {
        int B = Z->value.rows();
        Tensor sm = Tensor::softmax_row(Z->value);
        Tensor gZ = (sm - Y->value) * (1.0f / float(B));
        if (Z->requires_grad) Z->grad.add_( gZ );
        if (Y->requires_grad) {
            Tensor lse = Tensor::logsumexp_row(Z->value);
            Tensor lsm = Z->value - lse;
            Tensor gY  = lsm * (-1.0f / float(B));
            Y->grad.add_( gY );
        }
    } else {
        throw std::runtime_error("VJP for CeWithLogits on CUDA not implemented yet!");
    }
}

void vjp_KLDivergence(Node* n, const Tensor& gy){
    Node* Z = n->inputs[0].get();
    Node* Y = n->inputs[1].get();
    if (n->value.is_cpu()) {
        int B = Z->value.rows();
        Tensor sm = Tensor::softmax_row(Z->value);
        Tensor gZ = (sm - Y->value) * (1.0f / float(B));
        if (Z->requires_grad) Z->grad.add_( gZ );
        if (Y->requires_grad) {
            Tensor lse = Tensor::logsumexp_row(Z->value);
            Tensor lsm = Z->value - lse;
            Tensor gY = (Tensor::log(Y->value) + Tensor::ones_like(Y->value) - lsm) * (1.0f / float(B));
            Y->grad.add_( gY );
        }
    } else {
        throw std::runtime_error("VJP for KLDivergence on CUDA not implemented yet!");
    }
}

// ----- Other Math -----
void vjp_Div(Node* n, const Tensor& gy){
    Node* A = n->inputs[0].get(); Node* B = n->inputs[1].get();
    if (n->value.is_cpu()) {
        if (A->requires_grad) A->grad.add_( rt( gy * Tensor::reciprocal(B->value), A->value) );
        if (B->requires_grad) B->grad.add_( rt( -gy * Tensor::reciprocal(B->value) * Tensor::reciprocal(B->value) * A->value, B->value) );
    } else {
        throw std::runtime_error("VJP for Div on CUDA not implemented yet!");
    }
}
void vjp_Reciprocal(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;
    if (n->value.is_cpu()) {
        Tensor recip_x = Tensor::reciprocal(X->value);
        X->grad.add_( rt( -gy * recip_x * recip_x, X->value) );
    } else {
        throw std::runtime_error("VJP for Reciprocal on CUDA not implemented yet!");
    }
}

void vjp_Linear(Node* n, const Tensor& gy){
    // Assuming vjp_Linear is a composite and can be handled by its constituents (MatMul, Add)
    // For a truly fused kernel, this would need a device-aware implementation.
    vjp_FMA(n, gy);
}

void vjp_Cosh(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;
    if (n->value.is_cpu()) {
        X->grad.add_( rt( gy * Tensor::sinh(X->value), X->value) );
    } else {
        throw std::runtime_error("VJP for Cosh on CUDA not implemented yet!");
    }
}

void vjp_Sinh(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;
    if (n->value.is_cpu()) {
        X->grad.add_( rt( gy * Tensor::cosh(X->value), X->value) );
    } else {
        throw std::runtime_error("VJP for Sinh on CUDA not implemented yet!");
    }
}

void vjp_Sign(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;
    if (n->value.is_cpu()) {
        X->grad.add_( rt( gy * 0.0f, X->value) ); // Gradient of sign is 0 almost everywhere
    } else {
        throw std::runtime_error("VJP for Sign on CUDA not implemented yet!");
    }
}

void vjp_Cos(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;
    if (n->value.is_cpu()) {
        X->grad.add_( rt( -1.0 * gy* Tensor::sin(X->value), X->value) );
    } else {
        throw std::runtime_error("VJP for Cos on CUDA not implemented yet!");
    }
}

void vjp_Sin(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;
    if (n->value.is_cpu()) {
        X->grad.add_( rt( gy * Tensor::cos(X->value), X->value) );
    } else {
        throw std::runtime_error("VJP for Sin on CUDA not implemented yet!");
    }
}

// void vjp_Sqrt(Node* n, const Tensor& gy){
//     Node* X = n->inputs[0].get();
//     if (!X->requires_grad) return;
//     if (n->value.is_cpu()) {
//         X->grad.add_( rt(0.5f * gy * Tensor::reciprocal(n->value), X->value) );
//     } else {
//         throw std::runtime_error("VJP for Sqrt on CUDA not implemented yet!");
//     }
// }

    void vjp_Sqrt(Node* n, const Tensor& gy){
        Node* X = n->inputs[0].get();
        if (!X->requires_grad) return;

        if (X->value.is_cpu()) {
            auto fn = ag::kernels::cpu().sqrt_bwd_from_y;
            if (fn) {
                // --- NEW: Call the fast backward kernel ---
                // The kernel uses the forward pass output `y = sqrt(x)`, which is `n->value`.
                Tensor dX_temp = Tensor::zeros_like(X->value);
                fn(n->value.data(), gy.data(), dX_temp.data(), X->value.numel());
                X->grad.add_(dX_temp); // Accumulate gradient
            } else {
                // --- OLD: Fallback to generic C++ ---
                // Derivative is 0.5 / sqrt(x) = 0.5 / y
                X->grad.add_( rt(0.5f * gy / n->value, X->value) );
            }
        } else {
            // GPU path (when ready)
            throw std::runtime_error("Sqrt backward on CUDA not implemented yet!");
        }
    }

void vjp_Relumask(Node* n, const Tensor& gy){
    Node* X = n->inputs[0].get();
    if (!X->requires_grad) return;
    if (n->value.is_cpu()) {
        X->grad.add_( rt( gy * 0.0f, X->value) ); // Gradient is 0
    } else {
        throw std::runtime_error("VJP for Relumask on CUDA not implemented yet!");
    }
}

void vjp_RELUAtt(Node* n, const Tensor& gy){
    if (n->value.is_cpu()) {
        Node* A = n->inputs[0].get(), *B = n->inputs[1].get(), *C = n->inputs[2].get(), *D = n->inputs[3].get();
        Tensor q = *n->tape[0], k = *n->tape[1], v = *n->tape[2], s = *n->tape[3];
        float scale = 1.0f / std::sqrt(float(k.cols()));
        Tensor dL_ds = Tensor::matmul(gy, Tensor::transpose(v));
        Tensor dL_dv = Tensor::matmul(Tensor::transpose(s), gy);
        Tensor dL_dg = Tensor::relu_mask(s) * dL_ds;
        Tensor dL_dq = Tensor::matmul(dL_dg, k);
        Tensor dL_dk = Tensor::matmul(Tensor::transpose(dL_dg), q);
        Tensor dL_dA_q = Tensor::matmul(dL_dq, B->value) * scale;
        Tensor dL_dB   = Tensor::matmul(Tensor::transpose(dL_dq), A->value) * scale;
        Tensor dL_dA_k = Tensor::matmul(dL_dk, C->value) * scale;
        Tensor dL_dC   = Tensor::matmul(Tensor::transpose(dL_dk), A->value) * scale;
        Tensor dL_dA_v = Tensor::matmul(dL_dv, D->value);
        Tensor dL_dD   = Tensor::matmul(Tensor::transpose(dL_dv), A->value);
        if (A->requires_grad) A->grad.add_(dL_dA_q + dL_dA_k + dL_dA_v);
        if (B->requires_grad) B->grad.add_(dL_dB);
        if (C->requires_grad) C->grad.add_(dL_dC);
        if (D->requires_grad) D->grad.add_(dL_dD);
    } else {
        throw std::runtime_error("VJP for RELUAtt on CUDA not implemented yet!");
    }
}

void vjp_MOE(Node* n, const Tensor& gy){
    if (n->value.is_cpu()) {
        Node* X = n->inputs[0].get();
        Node* W = n->inputs[1].get();
        Node* B = n->inputs[2].get();
        Tensor dL_dB = gy;
        Tensor dL_dW = Tensor::matmul(Tensor::transpose(gy), X->value);
        Tensor dL_dX = Tensor::matmul(gy, W->value);
        if (X->requires_grad) X->grad.add_(dL_dX);
        if (W->requires_grad) W->grad.add_(dL_dW);
        if (B->requires_grad) B->grad.add_(dL_dB);
    } else {
        throw std::runtime_error("VJP for MOE on CUDA not implemented yet!");
    }
}

void vjp_SigAtt(Node* n, const Tensor& gy){
    if (n->value.is_cpu()) {
        Node* A = n->inputs[0].get(), *B = n->inputs[1].get(), *C = n->inputs[2].get(), *D = n->inputs[3].get();
        Tensor q = *n->tape[0], k = *n->tape[1], v = *n->tape[2], s = *n->tape[3];
        float scale = 1.0f / std::sqrt(float(k.cols()));
        Tensor dL_ds = Tensor::matmul(gy, Tensor::transpose(v));
        Tensor dL_dv = Tensor::matmul(Tensor::transpose(s), gy);
        Tensor dL_dg = (s * (Tensor::ones_like(s) - s)) * dL_ds;
        Tensor dL_dq = Tensor::matmul(dL_dg, k);
        Tensor dL_dk = Tensor::matmul(Tensor::transpose(dL_dg), q);
        Tensor dL_dA_q = Tensor::matmul(dL_dq, B->value) * scale;
        Tensor dL_dB   = Tensor::matmul(Tensor::transpose(dL_dq), A->value) * scale;
        Tensor dL_dA_k = Tensor::matmul(dL_dk, C->value) * scale;
        Tensor dL_dC   = Tensor::matmul(Tensor::transpose(dL_dk), A->value) * scale;
        Tensor dL_dA_v = Tensor::matmul(dL_dv, D->value);
        Tensor dL_dD   = Tensor::matmul(Tensor::transpose(dL_dv), A->value);
        if (A->requires_grad) A->grad.add_(dL_dA_q + dL_dA_k + dL_dA_v);
        if (B->requires_grad) B->grad.add_(dL_dB);
        if (C->requires_grad) C->grad.add_(dL_dC);
        if (D->requires_grad) D->grad.add_(dL_dD);
    } else {
        throw std::runtime_error("VJP for SigAtt on CUDA not implemented yet!");
    }
}

// ----- Loss Functions -----
void vjp_MSELoss(Node* n, const Tensor& gy){
    Node* Z = n->inputs[0].get();
    Node* Y = n->inputs[1].get();
    if (n->value.is_cpu()) {
        int N = Z->value.numel();
        Tensor diff = Z->value - Y->value;
        Tensor gZ = diff * (2.0f / float(N));
        Tensor gY = -diff * (2.0f / float(N));
        if (Z->requires_grad) Z->grad.add_(gZ);
        if (Y->requires_grad) Y->grad.add_(gY);
    } else {
        throw std::runtime_error("VJP for MSELoss on CUDA not implemented yet!");
    }
}

void vjp_MAELoss(Node* n, const Tensor& gy){
    Node* Z = n->inputs[0].get();
    Node* Y = n->inputs[1].get();
    if (n->value.is_cpu()) {
        int N = Z->value.numel();
        Tensor diff = Tensor::sign(Z->value - Y->value);
        Tensor gZ = diff * (1.0f / float(N));
        Tensor gY = -diff * (1.0f / float(N));
        if (Z->requires_grad) Z->grad.add_(gZ);
        if (Y->requires_grad) Y->grad.add_(gY);
    } else {
        throw std::runtime_error("VJP for MAELoss on CUDA not implemented yet!");
    }
}

void vjp_Leaf(Node*, const Tensor&){ /* no-op */ }

} // namespace detail


// -------- dispatch table --------
VjpFn vjp_lookup(Op op){
    switch(op){
#define OP(name, arity, str) case Op::name: return &detail::vjp_##name;
#include "ad/detail/ops.def"
#undef OP
        default: return nullptr;
    }
}

} // namespace ag