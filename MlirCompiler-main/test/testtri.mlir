module {
  func.func @main(%arg0: tensor<4096x4096xf64>, %arg1: tensor<4096x4096xi32>) -> tensor<4096x4096xf32> {
    %0 = nova.sin %arg1 : tensor<4096x4096xi32>
    %1 = nova.cos %0 : tensor<4096x4096xf32>
    %2 = nova.tan %1 : tensor<4096x4096xf32>
    %3 = nova.sinh %2 : tensor<4096x4096xf32>
    %4 = nova.cosh %3 : tensor<4096x4096xf32>
    %41 = nova.asin %4 : tensor<4096x4096xf32>
    %411 = nova.acos %41: tensor<4096x4096xf32>
    %412 = nova.atan %411 : tensor<4096x4096xf32>
    %413 = nova.asinh %412 : tensor<4096x4096xf32>
    %414 = nova.acosh %413 : tensor<4096x4096xf32>
    %415 = nova.atanh %414 : tensor<4096x4096xf32>
    return %415 : tensor<4096x4096xf32>
  }
}

module {
  func.func @main(%arg0: tensor<1x8x3xi32>) -> tensor<8x3xi32> {
 %1 = "tosa.const_shape"() <{values = dense<[8, 3]> : tensor<2xindex>}> : () -> !tosa.shape<2>
 %2 = "tosa.reshape"(%arg0, %1) : (tensor<1x8x3xi32>, !tosa.shape<2>) -> tensor<8x3xi32>
 return %2 :tensor<8x3xi32>
  }
}