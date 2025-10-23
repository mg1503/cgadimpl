// =====================
// file: cgadimpl/src/nodeops.cpp
/*
 *  *****************    ~~~coder guide~~~    *****************/
//
//
// example demonstration of how to use custom kernel implementations
// for node operations like add, relu, etc.
// You can implement these functions to call optimized CPU/GPU kernels
// instead of using the default tensor operations.
// This is especially useful for performance-critical applications.
// The following is a skeleton implementation showing where to integrate
// your custom kernels.
// something like the below can be done inside each node operation function.
//
//
// *******************************************************
// std::shared_ptr<Node> relu_nodeops(const std::shared_ptr<Node>& x) {
//   const Tensor& xin = x->value;ctest
//   Tensor y = Tensor::zeros_like(xin);
//
//   if (A.is_cpu()) ag::kernels::cpu().relu /*or add*/( /* your CPU args */ );
//   else            ag::kernels::cuda().relu /*or add*/( /* + current_stream() */ );
//
//   auto n = std::make_shared<Node>(y, x->requires_grad, Op::Relu, "relu");
//   n->inputs = { x };
//   return n;
// }
//

// *******************************************************/

// =====================
#include "ad/nodeops.hpp"
#include "ad/runtime.hpp"
#include "ad/kernels_api.hpp"
#include <cuda_runtime.h>


namespace ag {
namespace detail {


// std::shared_ptr<Node> add_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ 

//      const Tensor& A = a->value;
//          const Tensor& B = b->value;

//          auto [M,K]  = A.shape();
//          auto [K2,N] = B.shape();
//          if (K != K2) throw std::runtime_error("add: inner dims mismatch");


//          auto* fn = ag::kernels::cpu().add;
//          if (!fn) 
//          {
         
//         //  Tensor y = a->value + b->value; 
//         // auto n = std::make_shared<Node>(y, a->requires_grad || b->requires_grad, Op::Add, "+"); 
//         // n->inputs = {a, b}; 
//         // ag::debug::on_node_created(n); 
//                   throw std::runtime_error("No CPU Add kernel registered");

//         // return n; 

//          }
//                   Tensor C({M,N});

//          fn(A.data(), B.data(), C.data(), M*K);

//          auto n = std::make_shared<Node>(C,
//              (a->requires_grad || b->requires_grad),
//              Op::Add, "+");
//          n->inputs = { a, b };
//          return n;




//     }

// std::shared_ptr<Node> add_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ 
//         Tensor y = a->value + b->value; 
//         auto n = std::make_shared<Node>(y, a->requires_grad || b->requires_grad, Op::Add, "+"); 
//         n->inputs = {a, b}; 
//         ag::debug::on_node_created(n); 
//         return n; 
//     }
std::shared_ptr<Node> add_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){
    const Tensor& A = a->value;
    const Tensor& B = b->value;
    if (A.device() != B.device()) {
        throw std::runtime_error("add_nodeops: device mismatch between inputs.");
    }

    Tensor Y = Tensor::zeros_like(A); // Create output tensor on the same device

    if (A.is_cpu()) {
        Y = A + B; // Use the old CPU-based operator+
    } else {
        // Dispatch to the GPU kernel!
        ag::kernels::cuda().add(A.data(), B.data(), Y.data(), Y.numel(), ag::current_stream());
    }

    auto n = std::make_shared<Node>(Y, a->requires_grad || b->requires_grad, Op::Add, "+");
    n->inputs = {a, b};
    ag::debug::on_node_created(n);
    return n;
}
    // std::shared_ptr<Node> sub_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ 
 
    //  const Tensor& A = a->value;
    //      const Tensor& B = b->value;

    //      auto [M,K]  = A.shape();
    //      auto [K2,N] = B.shape();
    //      if (K != K2) throw std::runtime_error("sub: inner dims mismatch");


    //      auto* fn = ag::kernels::cpu().sub;
    //      if (!fn) 
    //      {
    //     //  Tensor y = a->value - b->value; 
    //     // auto n = std::make_shared<Node>(y, a->requires_grad || b->requires_grad, Op::Sub, "-"); 
    //     // n->inputs = {a, b}; 
    //     // ag::debug::on_node_created(n); 
    //               throw std::runtime_error("No CPU Sub kernel registered");

    //     // return n; 

    //      }
    //               Tensor C({M,N});

    //      fn(A.data(), B.data(), C.data(), M*K);

    //      auto n = std::make_shared<Node>(C,
    //          (a->requires_grad || b->requires_grad),
    //          Op::Sub, "-");
    //      n->inputs = { a, b };
    //      return n;


    // }

   std::shared_ptr<Node> sub_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ 
        Tensor y = a->value - b->value; 
        auto n = std::make_shared<Node>(y, a->requires_grad || b->requires_grad, Op::Sub, "-"); 
        n->inputs = {a, b}; 
        ag::debug::on_node_created(n); 
        return n; 
    }



    // std::shared_ptr<Node> mul_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ 
      
    //  const Tensor& A = a->value;
    //      const Tensor& B = b->value;

    //      auto [M,K]  = A.shape();
    //      auto [K2,N] = B.shape();
    //      if (K != K2) throw std::runtime_error("mul: inner dims mismatch");


    //      auto* fn = ag::kernels::cpu().hadmul;
    //      if (!fn) 
    //      {
    //     //  Tensor y = a->value - b->value; 
    //     // auto n = std::make_shared<Node>(y, a->requires_grad || b->requires_grad, Op::Sub, "-"); 
    //     // n->inputs = {a, b}; 
    //     // ag::debug::on_node_created(n); 
    //               throw std::runtime_error("No CPU Mul kernel registered");

    //     // return n; 

    //      }
    //               Tensor C({M,N});

    //      fn(A.data(), B.data(), C.data(), M*K);

    //      auto n = std::make_shared<Node>(C,
    //          (a->requires_grad || b->requires_grad),
    //          Op::Mul, "*");
    //      n->inputs = { a, b };
    //      return n;
    // }

    std::shared_ptr<Node> mul_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ 
        Tensor y = a->value * b->value; 
        auto n = std::make_shared<Node>(y, a->requires_grad || b->requires_grad, Op::Mul, "*"); 
        n->inputs = {a, b}; 
        ag::debug::on_node_created(n); 
        return n; 
    }

  std::shared_ptr<Node> flomul_nodeops(const std::shared_ptr<Node>& a, float b){ 
        auto c = std::make_shared<Node>(b*Tensor::ones_like(a->value), false, Op::Leaf, "leaf");
        Tensor y = a->value * c->value; 
        auto n = std::make_shared<Node>(y, a->requires_grad || c->requires_grad, Op::Mul, "*"); 
        n->inputs = {a, c}; 
        ag::debug::on_node_created(n); 
        return n; 
    }


//    std::shared_ptr<Node> relu_nodeops(const std::shared_ptr<Node>& x){ 
//         const Tensor& xin = x->value;
//         Tensor y = Tensor::zeros_like(xin);

//         auto* fn = ag::kernels::cpu().relu;
//         if (!fn) throw std::runtime_error("No CPU ReLU kernel registered");
//         fn(xin.data(), y.data(), static_cast<int64_t>(xin.numel()));

//         auto n = std::make_shared<Node>(y, x->requires_grad, Op::Relu, "relu");
//         n->inputs = { x };
//         return n;
//     }

// std::shared_ptr<Node> relu_nodeops(const std::shared_ptr<Node>& x){
//     const Tensor& X = x->value;
//     Tensor Y = Tensor::zeros_like(X);

//     if (X.is_cpu()) {
//         Y = Tensor::relu(X); // Use old CPU-based static method
//     } else {
//         // Dispatch to the GPU kernel!
//         ag::kernels::cuda().relu(X.data(), Y.data(), Y.numel(), ag::current_stream());
//     }

//     auto n = std::make_shared<Node>(Y, x->requires_grad, Op::Relu, "relu");
//     n->inputs = {x};
//     ag::debug::on_node_created(n);
//     return n;
// }

std::shared_ptr<Node> relu_nodeops(const std::shared_ptr<Node>& x){
    const Tensor& X = x->value;
    Tensor Y = Tensor::zeros_like(X);

    if (X.is_cpu()) {
        auto fn = ag::kernels::cpu().relu;
        if (fn) {
            // --- NEW: Call the fast AVX2 kernel ---
            fn(X.data(), Y.data(), X.numel());
        } else {
            // --- OLD: Fallback to generic C++ ---
            Y = Tensor::relu(X);
        }
    } else {
        // GPU path (when ready)
        // This will correctly dispatch to your existing CUDA ReLU kernel.
        auto fn = ag::kernels::cuda().relu;
        if (fn) {
            fn(X.data(), Y.data(), Y.numel(), ag::current_stream());
        } else {
            throw std::runtime_error("ReLU forward on CUDA not implemented or loaded.");
        }
    }

    auto n = std::make_shared<Node>(Y, x->requires_grad, Op::Relu, "relu");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}

// std::shared_ptr<Node> matmul_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
//     const Tensor& A = a->value;        // (M, K)
//     const Tensor& B = b->value;        // (K, N)
//     const int M = A.rows();
//     const int K = A.cols();
//     const int N = B.cols();

//     Tensor Y(M, N); // allocate output (CPU-only Tensor for now)

//     if (A.is_cpu()) {
//         ag::kernels::cpu().matmul(A.data(), B.data(), Y.data(), M, K, N);
//         std::cout<<"Using CPU MatMul kernel"<<std::endl;
//     } else {
//         ag::kernels::cuda().matmul(A.data(), B.data(), Y.data(), M, K, N,
//                                    ag::current_stream());
//         std::cout<<"Using CUDA MatMul kernel"<<std::endl;
//     }

//     auto n = std::make_shared<Node>(Y,
//                                     a->requires_grad || b->requires_grad,
//                                     Op::MatMul, "matmul");
//     n->inputs = {a, b};
//     ag::debug::on_node_created(n);
//     return n;
// }


// std::shared_ptr<Node> matmul_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
//     const Tensor& A = a->value;
//     const Tensor& B = b->value;
//     if (A.device() != B.device()) {
//         throw std::runtime_error("matmul_nodeops: device mismatch between inputs.");
//     }

//     Tensor C(A.rows(), B.cols(), A.device()); // Create output tensor on the same device

//     if (A.is_cpu()) {
//         C = Tensor::matmul(A, B); // Use old CPU-based static method
//     } else {
//         // Dispatch to the GPU kernel!  
//         ag::kernels::cuda().matmul(A.data(), B.data(), C.data(), A.rows(), A.cols(), B.cols(), ag::current_stream());
//     }

//     auto n = std::make_shared<Node>(C, a->requires_grad || b->requires_grad, Op::MatMul, "matmul");
//     n->inputs = {a, b};
//     ag::debug::on_node_created(n);
//     return n;
// }

std::shared_ptr<Node> matmul_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
    const Tensor& A = a->value;
    const Tensor& B = b->value;
    if (A.device() != B.device()) {
        throw std::runtime_error("matmul_nodeops: device mismatch between inputs.");
    }
    if (A.cols() != B.rows()) {
        throw std::runtime_error("matmul_nodeops: inner dimension mismatch.");
    }

    // Create the output tensor on the correct device.
    Tensor C = Tensor::zeros(A.rows(), B.cols(), A.device());

    if (A.is_cpu()) {
        auto fn = ag::kernels::cpu().matmul;
        if (fn) {
            // --- NEW: Call the fast AVX2/OpenMP kernel ---
            fn(A.data(), B.data(), C.data(), A.rows(), A.cols(), B.cols());
        } else {
            // --- OLD: Fallback to generic C++ ---
            // Note: We already zeroed C, so we must call the matmul that accumulates.
            // Or, we can just call the static method which creates a new tensor.
            C = Tensor::matmul(A, B); 
        }
    } else {
        // GPU path (This part is already correct and uses your working cuBLAS kernel)
        auto fn = ag::kernels::cuda().matmul;
        if (fn) {
            fn(A.data(), B.data(), C.data(), A.rows(), A.cols(), B.cols(), ag::current_stream());
        } else {
             throw std::runtime_error("MatMul forward on CUDA not implemented or loaded.");
        }
    }

    auto n = std::make_shared<Node>(C, a->requires_grad || b->requires_grad, Op::MatMul, "matmul");
    n->inputs = {a, b};
    ag::debug::on_node_created(n);
    return n;
}

    // std::shared_ptr<Node> fmab_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c){ 
    //     const Tensor& A = a->value;
    //      const Tensor& B = b->value;

    //      auto [M,K]  = A.shape();
    //      auto [K2,N] = B.shape();
    //      if (K != K2) throw std::runtime_error("gemm: inner dims mismatch");

    //      const Tensor& C = c->value; 

    //              Tensor E({M,N});
 

    //      auto* fn = ag::kernels::cpu().fmab;
    //      if (!fn) throw std::runtime_error("No CPU GEMM kernel registered now only");
    //      fn(A.data(), B.data(), C.data(), E.data(), M, K, N);

    //      auto n = std::make_shared<Node>(E,
    //          (a->requires_grad || b->requires_grad || c->requires_grad),
    //          Op::FMA, "fmab");
    //      n->inputs = { a, b , c};
    //      return n;
    // }
    std::shared_ptr<Node> fmab_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c){ 
        Tensor y = Tensor::matmul(a->value, b->value)+c->value; 
        auto n = std::make_shared<Node>(y, a->requires_grad || b->requires_grad || c->requires_grad, Op::FMA, "fmab"); 
        n->inputs = {a, b, c}; ag::debug::on_node_created(n); 
        return n; 
    }

// std::shared_ptr<Node> attention_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d){ 
//     Tensor q = Tensor::matmul(a->value, b->value); 
//     Tensor k = Tensor::matmul(a->value, c->value); 
//     Tensor v = Tensor::matmul(a->value, d->value);
//     Tensor g = Tensor::matmul(q, Tensor::transpose(k)*(1.f/sqrt(float(k.cols())))) ;
//     Tensor s = Tensor::softmax_row(g);


//     int B = 1;                  // or a->value.batch() if batched
//     int nh = 1;         // defined in your model config
//     int N = a->value.rows();    // sequence length
//     int x = b->value.cols();    // per-head hidden dim

//     // Allocate output tensor (same shape as q)
//     Tensor o({q.shape().first,q.shape().second});  

//     // Call the CUDA flash kernel
//     std::cout<<"Functionalaaa";


//     auto* fn = ag::kernels::cpu().flasha;
//             if (!fn) throw std::runtime_error("No CPU Flash Attention kernel registered now only");


//     fn(q.data(), k.data(), v.data(), o.data(),
//                     B, nh, N, x);

//     // Now wrap it in a Node as usual
//     auto n = std::make_shared<Node>(
//         o,
//         a->requires_grad || b->requires_grad || c->requires_grad || d->requires_grad,
//         Op::Attention,
//         "attention"
//         );        
    
    
    
    
//     n->inputs = {a, b, c, d};
//     n->tape.resize(4);
//     n->tape={std::make_shared<Tensor>(q), std::make_shared<Tensor>(k), std::make_shared<Tensor>(v), std::make_shared<Tensor>(s)};
//     ag::debug::on_node_created(n); 
//     return n; 
//     }


    std::shared_ptr<Node> attention_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d){ 
    Tensor q = Tensor::matmul(a->value, b->value); 
    Tensor k = Tensor::matmul(a->value, c->value); 
    Tensor v = Tensor::matmul(a->value, d->value);
    Tensor g = Tensor::matmul(q, Tensor::transpose(k)*(1.f/sqrt(float(k.cols())))) ;
    Tensor s = Tensor::softmax_row(g);
    Tensor y = Tensor::matmul(s, v);
    auto n = std::make_shared<Node>(y, a->requires_grad || b->requires_grad || c->requires_grad || d->requires_grad, Op::Attention, "attention"); 
    n->inputs = {a, b, c, d};
    n->tape.resize(4);
    n->tape={std::make_shared<Tensor>(q), std::make_shared<Tensor>(k), std::make_shared<Tensor>(v), std::make_shared<Tensor>(s)};
    ag::debug::on_node_created(n); 
    return n; 
    }


std::shared_ptr<Node> sigatt_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d){ 
    Tensor q = Tensor::matmul(a->value, b->value); 
    Tensor k = Tensor::matmul(a->value, c->value); 
    Tensor v = Tensor::matmul(a->value, d->value);
    Tensor g = Tensor::matmul(q, Tensor::transpose(k)*(1.f/sqrt(float(k.cols())))) ;
    Tensor s = Tensor::sigmoid(g);
    Tensor y = Tensor::matmul(s, v);
    auto n = std::make_shared<Node>(y, a->requires_grad || b->requires_grad || c->requires_grad || d->requires_grad, Op::SigAtt, "sigatt"); 
    n->inputs = {a, b, c, d};
    n->tape.resize(4);
    n->tape={std::make_shared<Tensor>(q), std::make_shared<Tensor>(k), std::make_shared<Tensor>(v), std::make_shared<Tensor>(s)};
    ag::debug::on_node_created(n); 
    return n; 
    }

std::shared_ptr<Node> reluatt_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d){ 
    Tensor q = Tensor::matmul(a->value, b->value); 
    Tensor k = Tensor::matmul(a->value, c->value); 
    Tensor v = Tensor::matmul(a->value, d->value);
    Tensor g = Tensor::matmul(q, Tensor::transpose(k)*(1.f/sqrt(float(k.cols())))) ;
    Tensor s = Tensor::relu(g);
    Tensor y = Tensor::matmul(s, v);
    auto n = std::make_shared<Node>(y, a->requires_grad || b->requires_grad || c->requires_grad || d->requires_grad, Op::RELUAtt, "reluatt"); 
    n->inputs = {a, b, c, d};
    n->tape.resize(4);
    n->tape={std::make_shared<Tensor>(q), std::make_shared<Tensor>(k), std::make_shared<Tensor>(v), std::make_shared<Tensor>(s)};
    ag::debug::on_node_created(n); 
    return n; 
    }

std::shared_ptr<Node> moewe_nodeops(const std::shared_ptr<Node>& x, const std::shared_ptr<Node>& w, const std::shared_ptr<Node>& b){ 
        Tensor y = Tensor::softmax_row(Tensor::matmul(x->value, Tensor::transpose(w->value)) + b->value); 
        auto n=std::make_shared<Node>(y, x->requires_grad, Op::MOE, "moe"); 
        n->inputs={x}; 
        ag::debug::on_node_created(n);  
        return n;
    }

// std::shared_ptr<Node> div_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ 
//            const Tensor& A = a->value;
//          const Tensor& B = b->value;

//          auto [M,K]  = A.shape();
//          auto [K2,N] = B.shape();
//          if (K != K2) throw std::runtime_error("sub: inner dims mismatch");


//          auto* fn = ag::kernels::cpu().div;
//          if (!fn) 
//          {
//         //  Tensor y = a->value - b->value; 
//         // auto n = std::make_shared<Node>(y, a->requires_grad || b->requires_grad, Op::Sub, "-"); 
//         // n->inputs = {a, b}; 
//         // ag::debug::on_node_created(n); 
//                   throw std::runtime_error("No CPU Sub kernel registered");

//         // return n; 

//          }
//                   Tensor C({M,N});

//          fn(A.data(), B.data(), C.data(), M*K);

//          auto n = std::make_shared<Node>(C,
//              (a->requires_grad || b->requires_grad),
//              Op::Div, "/");
//          n->inputs = { a, b };
//          return n;
//     }

std::shared_ptr<Node> div_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){
    const Tensor& A = a->value;
    const Tensor& B = b->value;

    if (A.shape() != B.shape()) {
        throw std::runtime_error("div: tensor shapes must match for element-wise division");
    }

    Tensor C = Tensor::zeros_like(A);

    // Direct implementation of element-wise division
    const float* a_data = A.data();
    const float* b_data = B.data();
    float* c_data = C.data();
    const int64_t num_elements = A.numel();

    for (int64_t i = 0; i < num_elements; ++i) {
        // Note: This does not explicitly handle division by zero.
        // For better numerical stability, you might want to add a small epsilon to the divisor.
        c_data[i] = a_data[i] / b_data[i];
    }

    auto n = std::make_shared<Node>(C,
        (a->requires_grad || b->requires_grad),
        Op::Div, "/");
    n->inputs = { a, b };
    return n;
}

        std::shared_ptr<Node> reci_nodeops(const std::shared_ptr<Node>& a){ 
        Tensor y = Tensor::ones_like(a->value)/a->value; 
        auto n = std::make_shared<Node>(y, a->requires_grad, Op::Reciprocal, "reciprocal"); 
        n->inputs = {a}; 
        ag::debug::on_node_created(n); 
        return n; 
    }

     std::shared_ptr<Node> flodiv_nodeops(float b , const std::shared_ptr<Node>& a){ 
        auto c = std::make_shared<Node>(b*Tensor::ones_like(a->value), false, Op::Leaf, "leaf");
        Tensor y = c->value / a->value; 
        auto n = std::make_shared<Node>(y, a->requires_grad || c->requires_grad, Op::Div, "/"); 
        n->inputs = {a, c}; 
        ag::debug::on_node_created(n); 
        return n; 
    }


     std::shared_ptr<Node> floadd_nodeops(float b , const std::shared_ptr<Node>& a){ 
        auto c = std::make_shared<Node>(b*Tensor::ones_like(a->value), false, Op::Leaf, "leaf");
        Tensor y = c->value + a->value; 
        auto n = std::make_shared<Node>(y, a->requires_grad || c->requires_grad, Op::Add, "+"); 
        n->inputs = {a, c}; 
        ag::debug::on_node_created(n); 
        return n; 
    }


//  std::shared_ptr<Node> relumask_nodeops(const std::shared_ptr<Node>& x){ 
//         const Tensor& xin = x->value;
//         Tensor y = Tensor::zeros_like(xin);

//         auto* fn = ag::kernels::cpu().relumask;
//         if (!fn) throw std::runtime_error("No CPU ReLU Mask kernel registered");
//         fn(xin.data(), y.data(), static_cast<int64_t>(xin.numel()));

//         auto n = std::make_shared<Node>(y, x->requires_grad, Op::Relumask, "relumask");
//         n->inputs = { x };
//         return n;
//     }

std::shared_ptr<Node> relumask_nodeops(const std::shared_ptr<Node>& x) {
    const Tensor& xin = x->value;
    Tensor y = Tensor::zeros_like(xin);

    // Direct implementation of the ReLU mask
    const float* x_data = xin.data();
    float* y_data = y.data();
    for (int64_t i = 0; i < xin.numel(); ++i) {
        if (x_data[i] > 0) {
            y_data[i] = 1.0f;
        }
    }

    auto n = std::make_shared<Node>(y, x->requires_grad, Op::Relumask, "relumask");
    n->inputs = {x};
    return n;
}


// std::shared_ptr<Node> linear_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c){ 
//         const Tensor& A = a->value;
//          const Tensor& B = Tensor::transpose(b->value);

//          auto [M,K]  = A.shape();
//          auto [K2,N] = B.shape();
//          if (K != K2) throw std::runtime_error("gemm: inner dims mismatch");

//          const Tensor& C = c->value; 

//                  Tensor E({M,N});


//          auto* fn = ag::kernels::cpu().fmab;
//          if (!fn) throw std::runtime_error("No CPU GEMM kernel registered now only");
//          fn(A.data(), B.data(), C.data(), E.data(), M, K, N);

//          auto n = std::make_shared<Node>(E,
//              (a->requires_grad || b->requires_grad || c->requires_grad),
//              Op::Linear, "linear");
//          n->inputs = { a, b , c};
//          return n;
//     }


std::shared_ptr<Node> linear_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c){
    const Tensor& A = a->value;
    const Tensor B = Tensor::transpose(b->value);

    auto [M,K]  = A.shape();
    auto [K2,N] = B.shape();
    if (K != K2) throw std::runtime_error("gemm: inner dims mismatch");

    const Tensor& C = c->value;

    Tensor E({M,N});

    // Direct implementation of fused multiply-add (A * B + C)
    const float* a_data = A.data();
    const float* b_data = B.data();
    const float* c_data = C.data();
    float* e_data = E.data();

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                // A is row-major, B (transposed) is row-major
                sum += a_data[i * K + k] * b_data[k * N + j];
            }
            // Add bias, assuming C can be broadcasted if it's a vector
            e_data[i * N + j] = sum + (C.numel() == N ? c_data[j] : c_data[i * N + j]);
        }
    }

    auto n = std::make_shared<Node>(E,
        (a->requires_grad || b->requires_grad || c->requires_grad),
        Op::Linear, "linear");
    n->inputs = { a, b , c};
    return n;
}


    std::shared_ptr<Node> cosh_nodeops(const std::shared_ptr<Node>& x){ 
        Tensor y = Tensor::cosh(x->value); 
        auto n=std::make_shared<Node>(y, x->requires_grad, Op::Cosh, "cosh"); 
        n->inputs={x}; 
        ag::debug::on_node_created(n);  
        return n;
    }

     std::shared_ptr<Node> sinh_nodeops(const std::shared_ptr<Node>& x){ 
        Tensor y = Tensor::sinh(x->value); 
        auto n=std::make_shared<Node>(y, x->requires_grad, Op::Sinh, "sinh"); 
        n->inputs={x}; 
        ag::debug::on_node_created(n);  
        return n;
    }


     std::shared_ptr<Node> cos_nodeops(const std::shared_ptr<Node>& x){ 
        Tensor y = Tensor::cosh(x->value); 
        auto n=std::make_shared<Node>(y, x->requires_grad, Op::Cosh, "cosh"); 
        n->inputs={x}; 
        ag::debug::on_node_created(n);  
        return n;
    }

     std::shared_ptr<Node> sin_nodeops(const std::shared_ptr<Node>& x){ 
        Tensor y = Tensor::sinh(x->value); 
        auto n=std::make_shared<Node>(y, x->requires_grad, Op::Sinh, "sinh"); 
        n->inputs={x}; 
        ag::debug::on_node_created(n);  
        return n;
    }


        std::shared_ptr<Node> sign_nodeops(const std::shared_ptr<Node>& x){ 
        Tensor y = Tensor::sign(x->value); 

        auto n=std::make_shared<Node>(y, x->requires_grad, Op::Sign, "sign"); 
        n->inputs={x}; 
        ag::debug::on_node_created(n);  
        return n;
    }

    //     std::shared_ptr<Node> sqrt_nodeops(const std::shared_ptr<Node>& x){ 
    //     Tensor y = Tensor::sqrt(x->value); 

    //     auto n=std::make_shared<Node>(y, x->requires_grad, Op::Sqrt, "sqrt"); 
    //     n->inputs={x}; 
    //     ag::debug::on_node_created(n);  
    //     return n;
    // }

std::shared_ptr<Node> sqrt_nodeops(const std::shared_ptr<Node>& x){
    const Tensor& X = x->value;
    Tensor Y = Tensor::zeros_like(X);

    if (X.is_cpu()) {
        auto fn = ag::kernels::cpu().sqrt;
        if (fn) {
            // --- NEW: Call the fast AVX2 kernel ---
            fn(X.data(), Y.data(), X.numel());
        } else {
            // --- OLD: Fallback to generic C++ ---
            Y = Tensor::sqrt(X);
        }
    } else {
        // GPU path (when ready)
        throw std::runtime_error("Sqrt forward on CUDA not implemented yet!");
    }

    auto n = std::make_shared<Node>(Y, x->requires_grad, Op::Sqrt, "sqrt");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}


    std::shared_ptr<Node> alibiatt_nodeops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d, float& m) { 
    Tensor q = Tensor::matmul(a->value, b->value); 
    Tensor k = Tensor::matmul(a->value, c->value); 
    Tensor v = Tensor::matmul(a->value, d->value);
    
    Tensor logits = Tensor::matmul(q, Tensor::transpose(k) * (1.f / sqrt(float(k.cols()))));
    Tensor bias   = Tensor::alibi(logits.rows(), logits.cols(), m);
    Tensor g      = logits + bias;

    Tensor s = Tensor::softmax_row(g);
    Tensor y = Tensor::matmul(s, v);

    auto n = std::make_shared<Node>(
        y, a->requires_grad || b->requires_grad || c->requires_grad || d->requires_grad,
        Op::AlibiAttention, "alibiattention"
    ); 
    n->inputs = {a, b, c, d};
    n->tape.resize(4);
    n->tape   = {std::make_shared<Tensor>(q), std::make_shared<Tensor>(k), 
                 std::make_shared<Tensor>(v), std::make_shared<Tensor>(s)};
    ag::debug::on_node_created(n); 
    return n; 
}


    std::shared_ptr<Node> swiglu_nodeops(const std::shared_ptr<Node>& x, const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d){ 
    Tensor y = Tensor::matmul(x->value, Tensor::transpose(a->value))+b->value; 
    debug::print_tensor("y",y);
    Tensor q = y*Tensor::sigmoid(y); 
    Tensor w = q*(Tensor::matmul(x->value, Tensor::transpose(c->value)) + d->value);
    auto n=std::make_shared<Node>(w, x->requires_grad || a->requires_grad || b->requires_grad || c->requires_grad || d->requires_grad, Op::SWIGLU, "swiglu"); 
    n->inputs={x, a, b, c, d};
    ag::debug::on_node_created(n); 
    return n;
    }


    std::shared_ptr<Node> sum_nodeops(const std::shared_ptr<Node>& x){ 
        Tensor y = Tensor::sum_all(x->value); 
        auto n = std::make_shared<Node>(y, x->requires_grad, Op::Sum, "sum"); 
        n->inputs = {x}; 
        ag::debug::on_node_created(n);  
        return n; 
    }

    std::shared_ptr<Node> transpose_nodeops(const std::shared_ptr<Node>& x){ 
        Tensor y = Tensor::transpose(x->value); 
        auto n=std::make_shared<Node>(y, x->requires_grad, Op::Transpose, "exp"); 
        n->inputs={x}; 
        ag::debug::on_node_created(n);  
        return n;
    }


    // std::shared_ptr<Node> exp_nodeops(const std::shared_ptr<Node>& x){ 
    //           const Tensor& xin = x->value;
    //     Tensor y = Tensor::zeros_like(xin);

    //     auto* fn = ag::kernels::cpu().exp;
    //     if (!fn) throw std::runtime_error("No CPU Exp kernel registered");
    //     fn(xin.data(), y.data(), static_cast<int64_t>(xin.numel()));

    //     auto n = std::make_shared<Node>(y, x->requires_grad, Op::Exp, "exp");
    //     n->inputs = { x };
    //     return n;
    // }

    // std::shared_ptr<Node> exp_nodeops(const std::shared_ptr<Node>& x){ 
    //     Tensor y = Tensor::exp(x->value); 
    //     auto n=std::make_shared<Node>(y, x->requires_grad, Op::Exp, "exp"); 
    //     n->inputs={x}; 
    //     ag::debug::on_node_created(n);  
    //     return n;
    // }


std::shared_ptr<Node> exp_nodeops(const std::shared_ptr<Node>& x){
    const Tensor& X = x->value;
    Tensor Y = Tensor::zeros_like(X);

    if (X.is_cpu()) {
        auto fn = ag::kernels::cpu().exp;
        if (fn) {
            // --- NEW: Call the fast AVX2 kernel ---
            fn(X.data(), Y.data(), X.numel());
        } else {
            // --- OLD: Fallback to generic C++ ---
            Y = Tensor::exp(X);
        }
    } else {
        // GPU path (when ready)
        throw std::runtime_error("Exp forward on CUDA not implemented yet!");
    }

    auto n = std::make_shared<Node>(Y, x->requires_grad, Op::Exp, "exp");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}

    
    // std::shared_ptr<Node> log_nodeops(const std::shared_ptr<Node>& x){ 
    //     Tensor y = Tensor::log(x->value); 
    //     auto n=std::make_shared<Node>(y, x->requires_grad, Op::Log, "log"); 
    //     n->inputs={x}; 
    //     ag::debug::on_node_created(n);  
    //     return n;
    // }
    std::shared_ptr<Node> log_nodeops(const std::shared_ptr<Node>& x){
    const Tensor& X = x->value;
    Tensor Y = Tensor::zeros_like(X);

    if (X.is_cpu()) {
        auto fn = ag::kernels::cpu().log;
        if (fn) {
            // --- NEW: Call the fast AVX2 kernel ---
            fn(X.data(), Y.data(), X.numel());
        } else {
            // --- OLD: Fallback to generic C++ ---
            Y = Tensor::log(X);
        }
    } else {
        // GPU path (when ready)
        throw std::runtime_error("Log forward on CUDA not implemented yet!");
    }

    auto n = std::make_shared<Node>(Y, x->requires_grad, Op::Log, "log");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}


    std::shared_ptr<Node> mish_nodeops(const std::shared_ptr<Node>& x){ 
        Tensor y = x->value * Tensor::tanh( Tensor::softplus(x->value) ); 
        auto n=std::make_shared<Node>(y, x->requires_grad, Op::Mish, "mish"); 
        n->inputs={x}; 
        ag::debug::on_node_created(n);  
        return n;
    }
    
    // std::shared_ptr<Node> tanh_nodeops(const std::shared_ptr<Node>& x){ 
    //     Tensor y = Tensor::tanh(x->value); 
    //     auto n=std::make_shared<Node>(y, x->requires_grad, Op::Tanh, "tanh"); 
    //     n->inputs={x}; 
    //     ag::debug::on_node_created(n);  
    //     return n;
    // }
    
    std::shared_ptr<Node> tanh_nodeops(const std::shared_ptr<Node>& x){
    const Tensor& X = x->value;
    Tensor Y = Tensor::zeros_like(X);

    if (X.is_cpu()) {
        auto fn = ag::kernels::cpu().tanh;
        if (fn) {
            // --- NEW: Call the fast AVX2 kernel ---
            fn(X.data(), Y.data(), X.numel());
        } else {
            // --- OLD: Fallback to generic C++ ---
            Y = Tensor::tanh(X);
        }
    } else {
        // GPU path (when ready)
        throw std::runtime_error("Tanh forward on CUDA not implemented yet!");
    }

    auto n = std::make_shared<Node>(Y, x->requires_grad, Op::Tanh, "tanh");
    n->inputs = {x};
    ag::debug::on_node_created(n);
    return n;
}
    // std::shared_ptr<Node> sigmoid_nodeops(const std::shared_ptr<Node>& x){ 
    //                   const Tensor& xin = x->value;
    //     Tensor y = Tensor::zeros_like(xin);

    //     auto* fn = ag::kernels::cpu().sigmoid;
    //     if (!fn) throw std::runtime_error("No CPU Sigmoid kernel registered");
    //     fn(xin.data(), y.data(), static_cast<int64_t>(xin.numel()));

    //     auto n = std::make_shared<Node>(y, x->requires_grad, Op::Sigmoid, "sigmoid");
    //     n->inputs = { x };
    //     return n;
    // }


    std::shared_ptr<Node> sigmoid_nodeops(const std::shared_ptr<Node>& x){
        const Tensor& X = x->value;
        Tensor Y = Tensor::zeros_like(X);

        if (X.is_cpu()) {
            auto fn = ag::kernels::cpu().sigmoid;
            if (fn) {
                // --- NEW: Call the fast AVX2 kernel ---
                fn(X.data(), Y.data(), X.numel());
            } else {
                // --- OLD: Fallback to generic C++ ---
                Y = Tensor::sigmoid(X); 
            }
        } else {
            // GPU path (when ready)
            throw std::runtime_error("Sigmoid forward on CUDA not implemented");
        }

        auto n = std::make_shared<Node>(Y, x->requires_grad, Op::Sigmoid, "sigmoid"); 
        n->inputs={x}; 
        ag::debug::on_node_created(n);  
        return n;
    }
    
    // std::shared_ptr<Node> softplus_nodeops(const std::shared_ptr<Node>& x){ 
    //     Tensor y = Tensor::softplus(x->value); 
    //     auto n=std::make_shared<Node>(y, x->requires_grad, Op::Softplus, "softplus"); 
    //     n->inputs={x}; 
    //     ag::debug::on_node_created(n);  
    //     return n;
    // }

  std::shared_ptr<Node> softplus_nodeops(const std::shared_ptr<Node>& x){
        const Tensor& X = x->value;
        Tensor Y = Tensor::zeros_like(X);

        if (X.is_cpu()) {
            auto fn = ag::kernels::cpu().softplus; // Note: 'softmax' in your teammate's file was a typo for softplus
            if (fn) {
                // --- NEW: Call the fast AVX2 kernel ---
                fn(X.data(), Y.data(), X.numel());
            } else {
                // --- OLD: Fallback to generic C++ ---
                Y = Tensor::softplus(X);
            }
        } else {
            // GPU path (when ready)
            throw std::runtime_error("Softplus forward on CUDA not implemented yet!");
        }

        auto n = std::make_shared<Node>(Y, x->requires_grad, Op::Softplus, "softplus");
        n->inputs = {x};
        ag::debug::on_node_created(n);
        return n;
    }

    std::shared_ptr<Node> gaus_nodeops(const std::shared_ptr<Node>& x){ 
        Tensor y = Tensor::exp(-1*x->value*x->value); 
        auto n=std::make_shared<Node>(y, x->requires_grad, Op::Gaus, "gaus"); 
        n->inputs={x}; 
        ag::debug::on_node_created(n);  
        return n;
    }
    
    std::shared_ptr<Node> gelu_nodeops(const std::shared_ptr<Node>& x){ 
        const Tensor& X = x->value;
        Tensor Y = Tensor::zeros_like(X);

        if (X.is_cpu()) {
            auto fn = ag::kernels::cpu().gelu;
            if (fn) {
                // --- NEW: Call the fast kernel ---
                fn(X.data(), Y.data(), X.numel());
            } else {
                // --- OLD: Fallback to generic C++ ---
                Y = Tensor::gelu_tanh(X); 
            }
        } else {
            // GPU path (when ready)
            throw std::runtime_error("GELU forward on CUDA not implemented");
        }

        auto n = std::make_shared<Node>(Y, x->requires_grad, Op::GELU, "gelu"); 
        n->inputs={x}; 
        return n;
    }



    std::shared_ptr<Node> gcu_nodeops(const std::shared_ptr<Node>& x){ 
        Tensor y = x->value * Tensor::cos(x->value);
        auto n=std::make_shared<Node>(y, x->requires_grad, Op::GCU, "gcu"); 
        n->inputs={x}; 
        ag::debug::on_node_created(n);  
        return n;
    }
    
    std::shared_ptr<Node> silu_nodeops(const std::shared_ptr<Node>& x){ 
        Tensor y = Tensor::sigmoid(x->value); 
        y = y * x->value; 
        auto n=std::make_shared<Node>(y, x->requires_grad, Op::SiLU, "silu"); 
        n->inputs={x}; 
        ag::debug::on_node_created(n);  
        return n;
    }

    std::shared_ptr<Node> parcon_nodeops(const std::shared_ptr<Node>& x){ 
        Tensor y = x->value*(2*Tensor::ones_like(x->value)-x->value); 

        auto n=std::make_shared<Node>(y, x->requires_grad, Op::Parcon, "parcon"); 
        n->inputs={x}; 
        ag::debug::on_node_created(n);  
        return n;
    }

    std::shared_ptr<Node> lisht_nodeops(const std::shared_ptr<Node>& x){ 
        Tensor y = x->value*Tensor::tanh(x->value); 

        auto n=std::make_shared<Node>(y, x->requires_grad, Op::Parcon, "parcon"); 
        n->inputs={x}; 
        ag::debug::on_node_created(n);  
        return n;
    }
    
    
    std::shared_ptr<Node> leaky_relu_nodeops(const std::shared_ptr<Node>& x, float alpha){ 
        const Tensor& X = x->value;
        Tensor Y = Tensor::zeros_like(X);

        if (X.is_cpu()) {
            auto fn = ag::kernels::cpu().leakyrelu;
            if (fn) {
                // --- NEW: Call the fast AVX2 kernel ---
                fn(X.data(), Y.data(), X.numel(), alpha);
            } else {
                // --- OLD: Fallback to generic C++ ---
                Y = Tensor::leaky_relu(X, alpha); 
            }
        } else {
            // GPU path (when ready)
            throw std::runtime_error("LeakyReLU forward on CUDA not implemented");
        }

        // This part remains the same. It correctly adds alpha to the graph
        // so the backward pass can find it.
        Tensor aT(1,1); aT(0,0)=alpha; auto aC = constant(aT, "alpha"); 
        auto n = std::make_shared<Node>(Y, x->requires_grad, Op::LeakyRelu, "leakyrelu");
        n->inputs = {x, aC.node}; 
        ag::debug::on_node_created(n);  
        return n;
    }


    std::shared_ptr<Node> rowsum_nodeops(const std::shared_ptr<Node>& x){ 
        Tensor y = Tensor::row_sum(x->value); 
        auto n=std::make_shared<Node>(y, x->requires_grad, Op::RowSum, "rowsum"); 
        n->inputs={x}; 
        ag::debug::on_node_created(n);  
        return n;
    }
    
    std::shared_ptr<Node> rowmax_nodeops(const std::shared_ptr<Node>& x){ 
        Tensor y = Tensor::row_max(x->value); 
        auto n=std::make_shared<Node>(y, x->requires_grad, Op::RowMax, "rowmax"); 
        n->inputs={x}; 
        ag::debug::on_node_created(n);  
        return n;
    }


    std::shared_ptr<Node> rms_nodeops(const std::shared_ptr<Node>& x){ 
        Tensor z = Tensor::row_sum(x->value*x->value) * (1.f/x->value.cols());
        Tensor q = Tensor::sqrt(z + 1e-8f);
        Tensor y = x->value / q;

        auto n = std::make_shared<Node>(y, x->requires_grad, Op::RMSNorm, "rmsnorm");
        n->tape.resize(2);
        n->tape[0] = std::make_shared<Tensor>(q); // denominator
        n->tape[1] = std::make_shared<Tensor>(y);   // normalized output
        n->inputs = {x};
        return n;
    }

    std::shared_ptr<Node> realrms_nodeops(const std::shared_ptr<Node>& x, float& g){ 
        Tensor z = Tensor::row_sum(x->value*x->value) * (1.f/x->value.cols());
        Tensor q = Tensor::sqrt(z + 1e-8f);
        Tensor y = (x->value) / q;
                std::shared_ptr<Node> G =  std::make_shared<Node>(g*Tensor::ones_like(y), false, Op::Leaf, "leaf");;

        auto n = std::make_shared<Node>(y*g, x->requires_grad || G->requires_grad, Op::RealRMSNorm, "realrmsnorm");
        n->tape.resize(2);
        n->tape[0] = std::make_shared<Tensor>(q); // denominator
        n->tape[1] = std::make_shared<Tensor>(y);   // normalized output
        n->inputs = {x, G};
        return n;
    }

    std::shared_ptr<Node> laynor_nodeops(const std::shared_ptr<Node>& x){ 
        Tensor y = Tensor::row_sum(x->value)*(1.f/x->value.cols()); 
      //  std::cout<<"q      "<<y<<std::endl;
        Tensor vrc = Tensor::row_sum(((x->value )- y)*((x->value )- y))*(1.f/x->value.cols());
      //  std::cout<<"q      "<<vrc<<std::endl;
        Tensor q = ((x->value )- y)/(Tensor::sqrt(vrc)+0.01);
        
        auto n=std::make_shared<Node>(q, x->requires_grad, Op::LayerNorm, "layernorm");
      //  debug::print_tensor("q",q);
        n->tape.resize(2);
        n->tape[0] = std::make_shared<Tensor>(vrc);
        n->tape[1] = std::make_shared<Tensor>(y);

        n->inputs={x}; 
        ag::debug::on_node_created(n);  
        return n;
    }

    std::shared_ptr<Node> relaynor_nodeops(const std::shared_ptr<Node>& x, float& b, float& g){ 
        Tensor y = Tensor::row_sum(x->value)*(1.f/x->value.cols()); 
      //  std::cout<<"q      "<<y<<std::endl;
        Tensor vrc = Tensor::row_sum(((x->value )- y)*((x->value )- y))*(1.f/x->value.cols());
      //  std::cout<<"q      "<<vrc<<std::endl;
        Tensor q = ((((x->value )- y)/(Tensor::sqrt(vrc)+0.01)))   ;
        Tensor qg = (q*g) + b;

        std::shared_ptr<Node> B = std::make_shared<Node>(b*Tensor::ones_like(qg), false, Op::Leaf, "leaf");;
        std::shared_ptr<Node> G = std::make_shared<Node>(g*Tensor::ones_like(qg), false, Op::Leaf, "leaf");;
        
        auto n=std::make_shared<Node>(q, x->requires_grad || B->requires_grad||G->requires_grad, Op::RealLayerNorm, "reallayernorm");
      //  debug::print_tensor("q",q);
        n->tape.resize(3);
        n->tape[0] = std::make_shared<Tensor>(vrc);
        n->tape[1] = std::make_shared<Tensor>(y);
        n->tape[2] = std::make_shared<Tensor>(q);

        n->inputs={x}; 
        ag::debug::on_node_created(n);  
        return n;
    }
    
    std::shared_ptr<Node> mean_all_nodeops(const std::shared_ptr<Node>& x){ 
        Tensor y = Tensor::mean_all(x->value); 
        auto n=std::make_shared<Node>(y, x->requires_grad, Op::MeanAll, "meanall"); 
        n->inputs={x}; 
        ag::debug::on_node_created(n);  
        return n;
    }

    std::shared_ptr<Node> dyntanh_nodeops(const std::shared_ptr<Node>& x, float& a, float& b, float& g){ 
        Tensor h = x->value*a;
        Tensor y = Tensor::tanh(h)*g + b; 
        std::shared_ptr<Node> A = std::make_shared<Node>(a*Tensor::ones_like(x->value), false, Op::Leaf, "a");
        std::shared_ptr<Node> B = std::make_shared<Node>(b*Tensor::ones_like(x->value), false, Op::Leaf, "b");
        std::shared_ptr<Node> G = std::make_shared<Node>(g*Tensor::ones_like(x->value), false, Op::Leaf, "g");
        auto n=std::make_shared<Node>(y, x->requires_grad|| A->requires_grad|| B->requires_grad||G->requires_grad, Op::MeanAll, "meanall"); 
        n->inputs={x, A, B, G}; 
        n->tape.push_back(std::make_shared<Tensor>(h));
        ag::debug::on_node_created(n);  
        return n;
    }
    
    std::shared_ptr<Node> softmax_row_nodeops(const std::shared_ptr<Node>& z){ 
        Tensor y = Tensor::softmax_row(z->value); 
        auto n=std::make_shared<Node>(y, z->requires_grad, Op::SoftmaxRow, "softmax_row"); 
        n->inputs={z}; 
        ag::debug::on_node_created(n);  
        return n;
    }
    
    std::shared_ptr<Node> logsumexp_row_nodeops(const std::shared_ptr<Node>& z){ 
        Tensor y = Tensor::logsumexp_row(z->value); 
        auto n=std::make_shared<Node>(y, z->requires_grad, Op::LogSumExpRow, "logsumexp_row"); 
        n->inputs={z}; 
        ag::debug::on_node_created(n);  
        return n;
    }


std::shared_ptr<Node> mambassm_nodeops(const std::shared_ptr<Node>& z, const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d){ 

        if (z->tape.size()==0) {

                    Tensor w = Tensor::matmul(z->value, b->value); 
                    Tensor q = Tensor::matmul(w, c->value);
                    Tensor y = (z->value* d->value)+q;
                    auto W = std::make_shared<Node>(w, false, Op::Leaf, "leaf");
        auto n=std::make_shared<Node>(y, W->requires_grad || z->requires_grad || a->requires_grad || b->requires_grad || c->requires_grad || d->requires_grad, Op::LogSumExpRow, "logsumexp_row"); 
        n->inputs={z, a, b, c, d, W}; 
            z->tape.push_back(std::make_shared<Tensor>(w));
            z->inputs.push_back(W);
                    ag::debug::on_node_created(n);  
                    std::cout<<"Initialized SSM state"<<std::endl;
        return n;
                }
                else
                {

        Tensor w = Tensor::matmul(z->value, b->value)+(z->inputs.back()->value); 
                    Tensor q = Tensor::matmul(w, c->value);
                    Tensor y = (z->value* d->value)+q;
                    auto W = std::make_shared<Node>(w, false, Op::Leaf, "leaf");
        auto n=std::make_shared<Node>(y,  W->requires_grad || z->requires_grad || a->requires_grad || b->requires_grad || c->requires_grad || d->requires_grad, Op::LogSumExpRow, "logsumexp_row"); 
        n->inputs={z, a, b, c, d, W}; 
        z->tape.push_back(std::make_shared<Tensor>(w));
            z->inputs.push_back(W);
                    ag::debug::on_node_created(n);  
                    std::cout<<"SSM step"<<std::endl;
        return n;
                }

        
    }



     std::shared_ptr<Node> cross_entropy_with_logits_nodeops(const std::shared_ptr<Node>& logits,const std::shared_ptr<Node>& onehot){
    // Stable CE = mean( -sum(onehot * (logits - logsumexp_row(logits))) )
        Tensor Z = logits->value;
        Tensor Y = onehot->value;
        Tensor LSE = Tensor::logsumexp_row(Z); // [B,1]
        Tensor log_sm = Z - LSE; // [B,C]
        Tensor prod = Y * log_sm; // [B,C]
        Tensor rs = Tensor::row_sum(prod); // [B,1]
        Tensor s = Tensor::sum_all(rs); // [1,1]
        Tensor out = Tensor::mean_all(rs * Tensor::ones_like(rs)); // mean over B (same as s/B)
        // We'll compute exact mean: s / B
        Tensor mean(1,1); mean(0,0) = s(0,0) / float(Z.rows());
        Tensor loss = Tensor::zeros(1,1); loss(0,0) = -mean(0,0);
        auto n = std::make_shared<Node>(loss, logits->requires_grad || onehot->requires_grad, Op::CeWithLogits, "ce_with_logits");
        n->inputs = {logits, onehot};
        ag::debug::on_node_created(n);  
        return n;
    }


    std::shared_ptr<Node> kldivergence_nodeops(const std::shared_ptr<Node>& logits,const std::shared_ptr<Node>& onehot){
    // Stable CE = mean( -sum(onehot * (logits - logsumexp_row(logits))) )
        Tensor Z = logits->value;
        Tensor Y = onehot->value;
        Tensor X = Tensor::log(Y + Tensor::ones_like(Y)*1e-10f); // add small std::shared_ptr<Node> to avoid log(0)
        Tensor LSE = Tensor::logsumexp_row(Z); // [B,1]
        Tensor log_sm = X- Z + LSE; // [B,C]
        Tensor prod = Y * log_sm; // [B,C]
        Tensor rs = Tensor::row_sum(prod); // [B,1]
        Tensor s = Tensor::sum_all(rs); // [1,1]
        Tensor out = Tensor::mean_all(rs * Tensor::ones_like(rs)); // mean over B (same as s/B)
        // We'll compute exact mean: s / B
        Tensor mean(1,1); mean(0,0) = s(0,0) / float(Z.rows());
        Tensor loss = Tensor::zeros(1,1); loss(0,0) = -mean(0,0);
        auto n = std::make_shared<Node>(loss, logits->requires_grad || onehot->requires_grad, Op::KLDivergence, "kldivergence");
        n->inputs = {logits, onehot};
        ag::debug::on_node_created(n);  
        return n;
    }

    std::shared_ptr<Node> mse_loss_nodeops(const std::shared_ptr<Node>& pred, const std::shared_ptr<Node>& target) {
    Tensor diff = pred->value - target->value;
    Tensor sq   = diff * diff;               // elementwise
    Tensor s    = Tensor::sum_all(sq);                   // scalar [1,1]
    int B = pred->value.shape().first, C = pred->value.shape().second;
    Tensor scale = Tensor::ones(1,1);
    scale(0,0) = 1.0f / float(B * C);
    Tensor loss = s * scale;                 // broadcast scalar
    auto n = std::make_shared<Node>(loss, pred->requires_grad || target->requires_grad, Op::MSELoss, "mseloss");
    n->inputs = {pred, target};
        ag::debug::on_node_created(n);  
    return n;                 // broadcast scalar
}


    std::shared_ptr<Node> mae_loss_nodeops(const std::shared_ptr<Node>& pred, const std::shared_ptr<Node>& target) {
    Tensor diff = pred->value - target->value;
    Tensor sq   = Tensor::abs(diff);               // elementwise
    Tensor s    = Tensor::sum_all(sq);                   // scalar [1,1]
    int B = pred->value.shape().first, C = pred->value.shape().second;
    Tensor scale = Tensor::ones(1,1);
    scale(0,0) = 1.0f / float(B * C);
    Tensor loss = s * scale;                 // broadcast scalar
    auto n = std::make_shared<Node>(loss, pred->requires_grad || target->requires_grad, Op::MAELoss, "maeloss");
    n->inputs = {pred, target};
        ag::debug::on_node_created(n);  
    return n;                 // broadcast scalar
}

    // Tensor forward_eval_node_impl(const std::shared_ptr<Node>& node) {
    //     if (!node) throw std::runtime_error("forward_eval_node: null node");
    //     switch (node->op) {
    //         case Op::Add: return node->inputs[0]->value + node->inputs[1]->value;
    //         case Op::Sub: return node->inputs[0]->value - node->inputs[1]->value;
    //         case Op::Mul: return node->inputs[0]->value * node->inputs[1]->value;
    //         case Op::MatMul: return Tensor::matmul(node->inputs[0]->value, node->inputs[1]->value);
    //         case Op::Relu: return Tensor::relu(node->inputs[0]->value);
    //         case Op::Sigmoid: return Tensor::sigmoid(node->inputs[0]->value);
    //         case Op::Tanh: return Tensor::tanh(node->inputs[0]->value);
    //         case Op::Exp: return Tensor::exp(node->inputs[0]->value);
    //         case Op::Log: return Tensor::log(node->inputs[0]->value);
    //         case Op::AlibiAttention: {
    //             const Tensor &a = node->inputs[0]->value;
    //             const Tensor &b = node->inputs[1]->value;
    //             const Tensor &c = node->inputs[2]->value;
    //             const Tensor &d = node->inputs[3]->value;
    //             Tensor q = Tensor::matmul(a, b);
    //             Tensor k = Tensor::matmul(a, c);
    //             Tensor v = Tensor::matmul(a, d);
    //             Tensor logits = Tensor::matmul(q, Tensor::transpose(k) * (1.f / sqrt(float(k.cols()))));
    //             Tensor bias   = Tensor::alibi(logits.rows(), logits.cols(), /*m*/128);
    //             Tensor g      = logits + bias;
    //             Tensor s      = Tensor::softmax_row(g);
    //             return Tensor::matmul(s, v);
    //         }
    //         case Op::Leaf:
    //             return node->value;
    //         default:
    //             if (!node->tape.empty()) {
    //                 return *(node->tape.back());
    //             }
    //             throw std::runtime_error("forward_eval_node: unsupported op for recompute");
    //     }
    // }

    } // namespace detail
    } // namespace ag