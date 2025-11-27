func.func @test_pow(%arg0: tensor<1xi1>, %arg1: tensor<1xi4>, %arg2: tensor<1xi8>, %arg3: tensor<1xi16>, %arg4: tensor<1xi32>, %arg5: tensor<1xi64>, %arg6: tensor<1xf16>, %arg7: tensor<1xbf16>, %arg8: tensor<1xf32>, %arg9: tensor<1xf64>) -> (tensor<1xi1>, tensor<1xi4>, tensor<1xi8>, tensor<1xi16>, tensor<1xi32>, tensor<1xi64>, tensor<1xf16>, tensor<1xbf16>, tensor<1xf32>, tensor<1xf64>) {
  // I1
  %0 = nova.pow %arg0, %arg0 : tensor<1xi1>, tensor<1xi1> -> tensor<1xi1>
  // I4
  %1 = nova.pow %arg1, %arg1 : tensor<1xi4>, tensor<1xi4> -> tensor<1xi4>
  // I8
  %2 = nova.pow %arg2, %arg2 : tensor<1xi8>, tensor<1xi8> -> tensor<1xi8>
  // I16
  %3 = nova.pow %arg3, %arg3 : tensor<1xi16>, tensor<1xi16> -> tensor<1xi16>
  // I32
  %4 = nova.pow %arg4, %arg4 : tensor<1xi32>, tensor<1xi32> -> tensor<1xi32>
  // I64
  %5 = nova.pow %arg5, %arg5 : tensor<1xi64>, tensor<1xi64> -> tensor<1xi64>
  // F16
  %6 = nova.pow %arg6, %arg6 : tensor<1xf16>, tensor<1xf16> -> tensor<1xf16>
  // BF16
  %7 = nova.pow %arg7, %arg7 : tensor<1xbf16>, tensor<1xbf16> -> tensor<1xbf16>
  // F32
  %8 = nova.pow %arg8, %arg8 : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
  // F64
  %9 = nova.pow %arg9, %arg9 : tensor<1xf64>, tensor<1xf64> -> tensor<1xf64>

  return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9 : tensor<1xi1>, tensor<1xi4>, tensor<1xi8>, tensor<1xi16>, tensor<1xi32>, tensor<1xi64>, tensor<1xf16>, tensor<1xbf16>, tensor<1xf32>, tensor<1xf64>
}