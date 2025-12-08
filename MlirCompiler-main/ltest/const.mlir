func.func @test_constants() -> (tensor<1xi1>, tensor<1xi4>, tensor<1xi8>, tensor<1xi16>, tensor<1xi32>, tensor<1xi64>, tensor<1xf16>, tensor<1xbf16>, tensor<1xf32>, tensor<1xf64>) {
  // I1 (Boolean)
  %c_i1 = nova.constant {value = dense<[true]> : tensor<1xi1>} : tensor<1xi1>

  // I4 (4-bit Integer)
  %c_i4 = nova.constant {value = dense<[1]> : tensor<1xi4>} : tensor<1xi4>

  // I8 (8-bit Integer)
  %c_i8 = nova.constant {value = dense<[1]> : tensor<1xi8>} : tensor<1xi8>

  // I16 (16-bit Integer)
  %c_i16 = nova.constant {value = dense<[1]> : tensor<1xi16>} : tensor<1xi16>

  // I32 (32-bit Integer)
  %c_i32 = nova.constant {value = dense<[1]> : tensor<1xi32>} : tensor<1xi32>

  // I64 (64-bit Integer)
  %c_i64 = nova.constant {value = dense<[1]> : tensor<1xi64>} : tensor<1xi64>

  // F16 (16-bit Float)
  %c_f16 = nova.constant {value = dense<[1.0]> : tensor<1xf16>} : tensor<1xf16>

  // BF16 (BFloat16)
  %c_bf16 = nova.constant {value = dense<[1.0]> : tensor<1xbf16>} : tensor<1xbf16>

  // F32 (32-bit Float)
  %c_f32 = nova.constant {value = dense<[1.0]> : tensor<1xf32>} : tensor<1xf32>

  // F64 (64-bit Float)
  %c_f64 = nova.constant {value = dense<[1.0]> : tensor<1xf64>} : tensor<1xf64>

  return %c_i1, %c_i4, %c_i8, %c_i16, %c_i32, %c_i64, %c_f16, %c_bf16, %c_f32, %c_f64 : tensor<1xi1>, tensor<1xi4>, tensor<1xi8>, tensor<1xi16>, tensor<1xi32>, tensor<1xi64>, tensor<1xf16>, tensor<1xbf16>, tensor<1xf32>, tensor<1xf64>
}
