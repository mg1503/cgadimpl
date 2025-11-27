module {
  func.func @main(%arg0: tensor<4x3x2xf32>,%arg1:tensor<4x3x2xi64> ) -> tensor<4x3x2xi64> {
 %a =tosa.cast %arg0 : (tensor<4x3x2xf32>) -> tensor<4x3x2xi64>
  return %a :tensor<4x3x2xi64>
  }
}