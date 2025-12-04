func.func @main('%'arg0: tensor<8x16xf32>) -> tensor<1xf32> {
  %v0 = nova.matmul %arg0, %arg1: tensor<8x16xf32>, tensor<16x10xf32>
  %v1 = nova.add %v0, %arg2: tensor<8x10xf32>, tensor<1x10xf32>
  %v2 = nova.unknown_op %v1: tensor<8x10xf32>
  return %v2 : tensor<1xf32>
}