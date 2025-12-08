func.func @test_abs(%arg0: tensor<1xi1>, %arg1: tensor<1xi4>, %arg2: tensor<1xi8>, %arg3: tensor<1xi16>, %arg4: tensor<1xi32>, %arg5: tensor<1xi64>, %arg6: tensor<1xf16>, %arg7: tensor<1xbf16>, %arg8: tensor<1xf32>, %arg9: tensor<1xf64>) -> (tensor<1xi1>, tensor<1xi4>, tensor<1xi8>, tensor<1xi16>, tensor<1xi32>, tensor<1xi64>, tensor<1xf16>, tensor<1xbf16>, tensor<1xf32>, tensor<1xf64>) {
  // I1
  %0 = nova.abs %arg0 : tensor<1xi1>
  // I4
  %1 = nova.abs %arg1 : tensor<1xi4>
  // I8
  %2 = nova.abs %arg2 : tensor<1xi8>
  // I16
  %3 = nova.abs %arg3 : tensor<1xi16>
  // I32
  %4 = nova.abs %arg4 : tensor<1xi32>
  // I64
  %5 = nova.abs %arg5 : tensor<1xi64>
  // F16
  %6 = nova.abs %arg6 : tensor<1xf16>
  // BF16
  %7 = nova.abs %arg7 : tensor<1xbf16>
  // F32
  %8 = nova.abs %arg8 : tensor<1xf32>
  // F64
  %9 = nova.abs %arg9 : tensor<1xf64>

  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9 : tensor<1xi1>, tensor<1xi4>, tensor<1xi8>, tensor<1xi16>, tensor<1xi32>, tensor<1xi64>, tensor<1xf16>, tensor<1xbf16>, tensor<1xf32>, tensor<1xf64>
}