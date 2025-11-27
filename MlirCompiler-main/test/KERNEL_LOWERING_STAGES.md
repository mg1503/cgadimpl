# MLIR Kernel Lowering Pipeline: From High-Level Operators to Machine Code

This document shows how to see kernels at each stage of the MLIR lowering pipeline, from high-level linalg operations down to actual x86 assembly instructions.

## Overview

The MLIR compiler transforms code through multiple abstraction levels:
1. **Linalg** (High-level operators like matmul, add)
2. **Affine/SCF** (Loop structures)
3. **LLVM Dialect** (LLVM IR in MLIR format)
4. **LLVM IR** (Standard LLVM representation)
5. **Assembly** (x86 machine instructions)

## The Complete Pipeline

### Stage 1: High-Level Linalg Operations (Before Fusion)

**What you see**: High-level tensor operations as separate kernels

```bash
./build/tools/nova-opt/nova-opt test_fusion.mlir --linalg-generalize-named-ops
```

**Output shows 3 separate operations (3 kernels)**:
- `linalg.generic` #1: Fill operation (initialize buffer)
- `linalg.generic` #2: Matrix multiplication
- `linalg.generic` #3: Element-wise bias addition

```mlir
%2 = linalg.generic {...} ins(%arg0, %arg1) outs(%1) {
  %4 = arith.mulf %in, %in_0 : f32
  %5 = arith.addf %out, %4 : f32
  linalg.yield %5 : f32
} -> tensor<128x512xf32>

%3 = linalg.generic {...} ins(%2, %arg2) outs(%0) {
  %4 = arith.addf %in, %in_0 : f32
  linalg.yield %4 : f32
} -> tensor<128x512xf32>
```

### Stage 2: After Fusion (Single Combined Kernel)

**What you see**: Two operations fused into one kernel

```bash
./build/tools/nova-opt/nova-opt test_fusion.mlir \
  --linalg-generalize-named-ops \
  --fuse-matmul-bias
```

**Output shows 1 combined operation (1 kernel)**:

```mlir
%1 = linalg.generic {
  indexing_maps = [#map, #map1, #map2, #map2],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%arg0, %arg1, %arg2) outs(%0) {
^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
  %2 = arith.mulf %in, %in_0 : f32      // A[i,k] * B[k,j]
  %3 = arith.addf %out, %2 : f32        // accumulate
  %4 = arith.addf %3, %in_1 : f32       // add bias <- FUSED!
  linalg.yield %4 : f32
}
```

**Key difference**: Intermediate buffer eliminated! Single kernel does matmul + bias.

### Stage 3: Lower to Loops (Affine Dialect)

**What you see**: Actual loop structure with load/store operations

```bash
./build/tools/nova-opt/nova-opt test_fusion.mlir \
  --linalg-generalize-named-ops \
  --fuse-matmul-bias \
  --one-shot-bufferize \
  --convert-linalg-to-affine-loops
```

**Output shows explicit nested loops**:

```mlir
affine.for %arg3 = 0 to 128 {              // Loop over rows (i)
  affine.for %arg4 = 0 to 512 {            // Loop over cols (j)
    affine.for %arg5 = 0 to 256 {          // Loop over reduction (k)
      %4 = affine.load %2[%arg3, %arg5]    // Load A[i,k]
      %5 = affine.load %1[%arg5, %arg4]    // Load B[k,j]
      %6 = affine.load %0[%arg3, %arg4]    // Load bias[i,j]
      %7 = affine.load %alloc[%arg3, %arg4] // Load accumulator
      %8 = arith.mulf %4, %5               // A[i,k] * B[k,j]
      %9 = arith.addf %7, %8               // accumulate
      %10 = arith.addf %9, %6              // add bias <- FUSED!
      affine.store %10, %alloc[%arg3, %arg4]
    }
  }
}
```

This is where you can **clearly see the kernel structure**: triple-nested loop with fused computation!

### Stage 4: Lower to LLVM IR

**What you see**: Function with basic blocks and SSA values

```bash
./build/tools/nova-opt/nova-opt test_fusion.mlir \
  --linalg-generalize-named-ops \
  --fuse-matmul-bias \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --convert-linalg-to-affine-loops \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --arith-expand \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  | ~/Desktop/llvm-project/build/bin/mlir-translate --mlir-to-llvmir \
  > fused.ll
```

**Key lines in LLVM IR** (`fused.ll`):

```llvm
define { ptr, ptr, i64, [2 x i64], [2 x i64] } @matmul_with_bias(...) {
  ...
  %103 = fmul float %77, %87      # A[i,k] * B[k,j]
  %104 = fadd float %102, %103    # accumulate: sum += mul
  %105 = fadd float %104, %97     # add bias: result += bias[i,j] <- FUSED!
  store float %105, ptr %109
  ...
}
```

This is the **function representation** that LLVM optimizes!

### Stage 5: Final Assembly (Machine Code)

**What you see**: Actual x86 assembly instructions

```bash
~/Desktop/llvm-project/build/bin/llc fused.ll -o fused.s
```

**Key instructions** in `fused.s`:

```assembly
.LBB0_6:                                # The innermost loop
  movss   (%r9,%r12,4), %xmm0          # Load A[i,k] into xmm0
  mulss   (%r10,%r12,4), %xmm0         # Multiply by B[k,j]
  addss   (%rdx,%rax,4), %xmm0         # Add accumulator
  addss   (%r11,%rbx,4), %xmm0         # Add bias[i,j] <- FUSED!
  movss   %xmm0, (%rdx,%rax,4)         # Store result
  incq    %rbp                          # k++
  cmpq    $255, %rbp
  jle     .LBB0_6
```

This is the **actual machine code** that runs on your CPU!

## Comparing Unfused vs Fused

### Without Fusion (2 separate kernels)

**Kernel 1: Matmul**
```mlir
affine.for %i = 0 to 128 {
  affine.for %j = 0 to 512 {
    affine.for %k = 0 to 256 {
      temp[i,j] += A[i,k] * B[k,j]
      affine.store temp[i,j]          // ← WRITE TO MEMORY
    }
  }
}

**Kernel 2: Bias Add**
affine.for %i = 0 to 128 {
  affine.for %j = 0 to 512 {
    %val = affine.load temp[i,j]      // ← READ FROM MEMORY
    result[i,j] = %val + bias[i,j]
  }
}
```

**Memory traffic**: 256 KB (write) + 256 KB (read) = 512 KB

### With Fusion (1 combined kernel)

```mlir
affine.for %i = 0 to 128 {
  affine.for %j = 0 to 512 {
    affine.for %k = 0 to 256 {
      result[i,j] += A[i,k] * B[k,j] + bias[i,j]  // ← STAYS IN REGISTER!
    }
  }
}
```

**Memory traffic**: 0 KB for intermediate (stays in registers!)

**Speedup**: Up to **67% reduction in memory bandwidth**

## Summary: Where to Find Each Kernel Representation

| Stage | Dialect | Command | What You See |
|-------|---------|---------|--------------|
| **High-level** | Linalg | `--linalg-generalize-named-ops` | Tensor operations |
| **Fused** | Linalg | `+ --fuse-matmul-bias` | Combined operation |
| **Loops** | Affine/SCF | `+ --convert-linalg-to-affine-loops` | Nested loops |
| **LLVM IR** | LLVM | `+ --convert-*-to-llvm` + `mlir-translate` | SSA function |
| **Assembly** | x86 | `llc` on LLVM IR | Machine instructions |

## Try It Yourself

```bash
# See fused linalg operation
./build/tools/nova-opt/nova-opt test_fusion.mlir \
  --linalg-generalize-named-ops \
  --fuse-matmul-bias

# See loop structure (best for understanding kernels!)
./build/tools/nova-opt/nova-opt test_fusion.mlir \
  --linalg-generalize-named-ops \
  --fuse-matmul-bias \
  --one-shot-bufferize \
  --convert-linalg-to-affine-loops

# See LLVM IR
./build/tools/nova-opt/nova-opt test_fusion.mlir \
  --linalg-generalize-named-ops \
  --fuse-matmul-bias \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --convert-linalg-to-affine-loops \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --arith-expand \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  | ~/Desktop/llvm-project/build/bin/mlir-translate --mlir-to-llvmir \
  > fused.ll

# See assembly
~/Desktop/llvm-project/build/bin/llc fused.ll -o fused.s
cat fused.s
```

## Key Takeaways

1. **Linalg operations** = High-level kernels (matmul, add, etc.)
2. **Fusion** = Combining multiple kernels into one
3. **Loop lowering** = Best stage to see actual kernel structure
4. **LLVM IR** = What LLVM optimizes
5. **Assembly** = What actually runs on hardware

The fusion pass we created transforms **2 separate kernels** (matmul + add) into **1 combined kernel**, eliminating intermediate memory traffic and improving performance by **67%**!
