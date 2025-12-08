# Kernel Fusion Pass Demo

## Test File: `test_fusion.mlir`

```mlir
func.func @matmul_with_bias(%A, %B, %bias) -> tensor {
  // Step 1: Matrix multiplication (128x256) × (256x512) = (128x512)
  %matmul = linalg.matmul ins(%A, %B) outs(%init)

  // Step 2: Add bias (element-wise)
  %result = linalg.add ins(%matmul, %bias) outs(%init)

  return %result
}
```

## Running the Fusion Pass

### Command:
```bash
./build/tools/nova-opt/nova-opt test_fusion.mlir \
  --linalg-generalize-named-ops \
  --fuse-matmul-bias
```

### Output:
```
=== Running FuseMatmulBias Pass ===
Found fusible matmul + add pattern!  ← YOUR PASS DETECTED IT!
=== FuseMatmulBias Pass Complete ===
```

## What the Pass Detected

### Pattern Found:
```mlir
// %2 = matmul (producer)
%2 = linalg.generic {
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%arg0, %arg1) {
  %mul = arith.mulf %in, %in_0
  %add = arith.addf %out, %mul
  linalg.yield %add
}

// %3 = add (consumer) - uses %2
%3 = linalg.generic {
  iterator_types = ["parallel", "parallel"]  ← All parallel = element-wise
} ins(%2, %arg2) {                           ← Uses matmul result
  %add = arith.addf %in, %in_0
  linalg.yield %add
}
```

### Why This is Fusible:
1. ✅ Matmul (%2) has ONE use (the add operation)
2. ✅ Add (%3) is element-wise (all parallel loops)
3. ✅ Producer-consumer relationship (matmul → add)

## Performance Impact (When Fused)

### Before Fusion (Current):
```
1. Compute matmul → write 128×512 = 65,536 floats to memory
2. Read 65,536 floats from memory
3. Add bias → write 65,536 floats to memory
Total memory traffic: 65K writes + 65K reads = 196 KB
```

### After Fusion (TODO: Implement):
```
1. Compute matmul + add bias in single pass
2. Write result directly
Total memory traffic: 65K writes = 65 KB
```

**Savings: 67% less memory traffic!**

## Comparison with MLIR Built-in Passes

### Built-in `--linalg-fuse-elementwise-ops`:
- Only fuses element-wise operations (parallel loops only)
- Cannot fuse matmul (has reduction) with add
- Limited to specific patterns

### Your Custom `--fuse-matmul-bias`:
- Specifically targets matmul + bias pattern (common in ML)
- Detects producer-consumer relationships
- Can be extended to other ML-specific patterns

## Next Steps

To complete the fusion, you need to implement the rewrite in `FuseMatmulBias.cpp`:

```cpp
// Currently at line 40:
return failure();  // Don't rewrite yet

// Should be:
// Create a new fused operation that:
// 1. Takes inputs: A, B, bias
// 2. Computes: C[i,j] = sum_k(A[i,k] * B[k,j]) + bias[i,j]
// 3. Returns fused result
```

## Other Fusible Patterns to Add

1. **Conv + ReLU**: `conv → max(0, x)`
2. **Conv + BatchNorm**: Fold BatchNorm into conv weights
3. **Matmul + ReLU**: `matmul → max(0, x)`
4. **Element-wise chains**: `mul → add → relu`

Each pattern follows the same structure:
1. Detect pattern in `matchAndRewrite`
2. Check constraints (single use, compatible types)
3. Create fused operation
4. Replace uses

## Files Created

- ✅ `test_fusion.mlir` - Test case with matmul + bias
- ✅ `lib/Transforms/FuseMatmulBias.cpp` - Fusion pass (detection works!)
- ✅ `include/Compiler/Transforms/FuseMatmulBias.h` - Header
- ✅ Pass registered in `Passes.td`
- ✅ Pass available via `--fuse-matmul-bias`

## Success!

Your fusion pass successfully detects the matmul + bias pattern. This is the foundation for building a high-performance ML compiler!
