module {
  func.func @matmul_with_bias(%A: tensor<4096x4096xf32>, 
                              %B: tensor<4096x4096xf32>,
                              %bias: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %init = tensor.empty() : tensor<4096x4096xf32>
    %C = linalg.fill ins(%cst : f32) outs(%init : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    
    // MATMUL
    %C_result = linalg.matmul ins(%A, %B : tensor<4096x4096xf32>, tensor<4096x4096xf32>)
                             outs(%C : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    
    // ADD BIAS (separate operation)
    %bias_init = tensor.empty() : tensor<4096x4096xf32>
    %result = linalg.add ins(%C_result, %bias : tensor<4096x4096xf32>, tensor<4096x4096xf32>)
                         outs(%bias_init : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    
    return %result : tensor<4096x4096xf32>
  }
}
