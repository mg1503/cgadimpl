module {
  // Declare external timing functions
  func.func private @get_time() -> i64
  func.func private @print_gflops(%m: i64, %n: i64, %k: i64, %time_us: i64)
  
  func.func @matmul_4096(%A: tensor<4096x4096xf32>, 
                         %B: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %init = tensor.empty() : tensor<4096x4096xf32>
    %C = linalg.fill ins(%cst : f32) outs(%init : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    
    %start = call @get_time() : () -> i64
    %result = linalg.matmul ins(%A, %B : tensor<4096x4096xf32>, tensor<4096x4096xf32>)
                           outs(%C : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %end = call @get_time() : () -> i64
    
    %duration = arith.subi %end, %start : i64
    %m = arith.constant 4096 : i64
    %n = arith.constant 4096 : i64
    %k = arith.constant 4096 : i64
    call @print_gflops(%m, %n, %k, %duration) : (i64, i64, i64, i64) -> ()
    
    return %result : tensor<4096x4096xf32>
  }

  func.func @main() -> i32 {
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 2.000000e+00 : f32
    %expected = arith.constant 8192.000000e+00 : f32  // 4096 * 2.0
    
    %A = tensor.splat %cst_0 : tensor<4096x4096xf32>
    %B = tensor.splat %cst_1 : tensor<4096x4096xf32>
    
    // Actual timed run
    %result = call @matmul_4096(%A, %B) : (tensor<4096x4096xf32>, tensor<4096x4096xf32>) -> tensor<4096x4096xf32>

    // Verify correctness
    %c0 = arith.constant 0 : index
    %first = tensor.extract %result[%c0, %c0] : tensor<4096x4096xf32>
    %is_correct = arith.cmpf "oeq", %first, %expected : f32
    
    %success = arith.constant 0 : i32
    %failure = arith.constant 1 : i32
    %ret = arith.select %is_correct, %success, %failure : i32
    return %ret : i32
  }
}
