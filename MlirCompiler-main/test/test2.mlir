module {
// Declare external timing functions (used by the runner)
func.func private @get_time() -> i64
func.func private @print_gflops(i64, i64, i64, i64)

// =========================================================================
// 1. Core Operation (Nova Dialect)
// =========================================================================

func.func @core_op(%A: tensor<1024x1024xf32>,
%B: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {

%0 = tensor.empty() : tensor<1024x1024xf32>
%zero = arith.constant 0.0 : f32
%2 = linalg.fill ins(%zero : f32) outs(%0 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
// --- Start Timing ---
%start = call @get_time() : () -> i64

// 1. Nova MatMul: C = A * B
// NOTE: For pure MatMul, the result C MUST be initialized to zero for a correct reduction.

%1 = linalg.matmul ins(%A, %B : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%2 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>


// --- End Timing ---
%end = call @get_time() : () -> i64

// Calculate and print GFLOPS
%duration = arith.subi %end, %start : i64
%m = arith.constant 1024 : i64
%n = arith.constant 1024 : i64
%k = arith.constant 1024 : i64
call @print_gflops(%m, %n, %k, %duration) : (i64, i64, i64, i64) -> ()

return %1 : tensor<1024x1024xf32>


}

// =========================================================================
// 2. Main Execution and Verification
// =========================================================================

func.func @main() -> i32 {
// Inputs: A=1.0, B=2.0
%c1 = arith.constant 1.0 : f32
%c2 = arith.constant 2.0 : f32
%c0 = arith.constant 0.0 : f32

%A = tensor.splat %c1 : tensor<1024x1024xf32>
%B = tensor.splat %c2 : tensor<1024x1024xf32>

// Expected: (1024 * 1.0 * 2.0) = 2048.0
%expected_val = arith.constant 2048.0 : f32

%result = call @core_op(%A, %B) : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

// --- Verification (checks element at [0, 0]) ---
%idx0 = arith.constant 0 : index
%first_element = tensor.extract %result[%idx0, %idx0] : tensor<1024x1024xf32>

// Tolerance of 0.001
%tolerance = arith.constant 0.001 : f32 
%diff = arith.subf %first_element, %expected_val : f32
%abs_diff = math.absf %diff : f32
%is_incorrect = arith.cmpf "oge", %abs_diff, %tolerance : f32
%ret = arith.extui %is_incorrect : i1 to i32

return %ret : i32


}
}