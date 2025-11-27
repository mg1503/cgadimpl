
  func.func @main(%arg0: tensor<3x3xi32>, %arg1: tensor<3x3xf64>) -> tensor<3x3xf64> {
    %a = nova.max %arg0, %arg1: tensor<3x3xi32>, tensor<3x3xf64>
    %b=nova.add %a,%arg0:tensor<3x3xf64>,tensor<3x3xi32>
    return %b : tensor<3x3xf64>
  }
