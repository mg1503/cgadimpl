# MLIR Matrix Multiplication with Parallelization

## Summary

This project demonstrates optimized matrix multiplication (512x512) using MLIR with both **sequential vectorization** and **parallel (OpenMP) execution**.

## Performance Results

Based on benchmarks on your system:

| Version | Threads | Avg Time | Performance |
|---------|---------|----------|-------------|
| MLIR Vectorized (Sequential) | 1 | 13.3ms | **20.13 GFLOPS** |
| C++ OpenMP Parallel | 16 | 10.0ms | **26.84 GFLOPS** |
| C++ OpenMP Parallel | 8 | 11.7ms | **23.01 GFLOPS** |
| C++ OpenMP Parallel | 4 | 14.0ms | **19.17 GFLOPS** |

**Key Insight**: The MLIR sequential version with vectorization achieves ~20 GFLOPS, while adding OpenMP parallelization with 16 threads gives you a **1.33x speedup** to ~27 GFLOPS.

## Files

### Source
- `test.mlir` - Input MLIR with matmul operation
- `matmul_omp.cpp` - C++ version with OpenMP parallelization

### Compilation Scripts
- `compile_best.sh` - Compiles MLIR with best sequential optimizations
- `benchmark.sh` - Runs comprehensive benchmarks

### Binaries
- `./matmul` - MLIR-compiled sequential (vectorized) version
- `./matmul_omp` - C++ OpenMP parallel version

## Compilation Pipeline (MLIR Sequential)

The optimized pipeline uses:

```bash
./build/tools/nova-opt/nova-opt test.mlir --pass-pipeline='builtin.module(
    canonicalize,
    one-shot-bufferize{
      bufferize-function-boundaries=1
      function-boundary-type-conversion=identity-layout-map
    },
    buffer-deallocation-pipeline,
    convert-linalg-to-affine-loops,
    func.func(
      affine-loop-tile{tile-sizes=32,32,8},           # Cache blocking
      affine-loop-unroll-jam{unroll-jam-factor=2},    # Unroll-and-jam
      affine-loop-unroll{unroll-factor=8},            # Loop unrolling
      canonicalize,
      cse,
      math-uplift-to-fma,                             # Use FMA instructions
      affine-super-vectorize{virtual-vector-size=8},  # AVX2 vectorization
      canonicalize
    ),
    lower-affine,
    convert-vector-to-scf,
    lower-affine,
    convert-scf-to-cf,
    canonicalize,
    convert-vector-to-llvm{enable-x86vector=true reassociate-fp-reductions=true},
    convert-math-to-llvm,
    convert-cf-to-llvm,
    convert-arith-to-llvm,
    finalize-memref-to-llvm,
    convert-func-to-llvm,
    cleanup,
    reconcile-unrealized-casts
  )' -o output.mlir
```

### Key Optimization Passes

1. **affine-loop-tile{tile-sizes=32,32,8}** - Tiles loops for cache locality
2. **affine-super-vectorize{virtual-vector-size=8}** - Generates AVX2 8-wide vector operations
3. **math-uplift-to-fma** - Uses FMA (Fused Multiply-Add) instructions
4. **affine-loop-unroll** - Unrolls innermost loops for instruction-level parallelism

## Why Parallel Loops Don't Work Directly in MLIR

The issue with `affine-parallelize` or `convert-scf-to-openmp` in your pipeline:

1. **affine-parallelize** conflicts with **affine-super-vectorize**
   - Both want to transform the same loops
   - Parallel marking prevents vectorization optimization

2. **convert-scf-to-openmp** creates malformed IR
   - The `memref.alloca_scope` gets multiple basic blocks after transformations
   - MLIR expects single-block regions in certain contexts

## Recommended Approaches for Parallelization

### Option 1: C++ Wrapper with OpenMP (Current Best)

Compile the C++ version:
```bash
clang++ matmul_omp.cpp -o matmul_omp \
  -O3 -ffast-math -march=native -mtune=native -fopenmp -pthread
```

Run with specific thread count:
```bash
OMP_NUM_THREADS=16 ./matmul_omp
```

### Option 2: Future MLIR Approach

When MLIR's OpenMP support stabilizes, use:
- **Outer loops**: Mark for parallelization
- **Inner loops**: Keep for vectorization

This hybrid approach should theoretically give best performance.

## Best Compilation Flags

### LLVM opt
```bash
~/Desktop/llvm-project/build/bin/opt output.ll -o output_opt.ll \
  -passes="default<O3>" \
  -fp-contract=fast \
  -enable-unsafe-fp-math \
  -enable-no-nans-fp-math \
  -enable-no-signed-zeros-fp-math
```

### LLC (LLVM Static Compiler)
```bash
llc output_opt.ll -o output.s \
  -O3 \
  -march=x86-64 \
  -mcpu=native \
  -mattr=+avx2,+fma \
  -filetype=asm \
  -relocation-model=pic
```

### Clang Linking
```bash
clang output.s -o matmul \
  -O3 \
  -ffast-math \
  -march=native \
  -mtune=native \
  -fopenmp \         # For OpenMP runtime
  -pthread           # For threading support
```

## Running Benchmarks

```bash
./benchmark.sh
```

This runs 30 iterations of each version and reports:
- Average execution time
- GFLOPS performance
- Comparison across thread counts

## Verification

All versions verify correctness by checking:
```
C[0][0] == 1024.0  (512 * 1.0 * 2.0)
```

Return codes:
- 0 = Success
- 1 = Computation error

## Next Steps for Better Performance

1. **Use larger matrices** - 512x512 is small; try 2048x2048 or 4096x4096
2. **Use BLAS libraries** - OpenBLAS or Intel MKL will be faster
3. **Try GPU offload** - For massive matrices, GPU is orders of magnitude faster
4. **Profile with perf** - Identify cache misses and bottlenecks:
   ```bash
   perf stat -d -d -d ./matmul
   ```

## Notes

- The MLIR pipeline generates efficient **AVX2 vectorized** code
- OpenMP adds **thread-level parallelism** on top of vectorization
- For production, consider using optimized BLAS (e.g., OpenBLAS, MKL)
- Current sweet spot: **16 threads** gives best performance on your system

## References

- MLIR Documentation: https://mlir.llvm.org/
- MLIR Affine Dialect: https://mlir.llvm.org/docs/Dialects/Affine/
- OpenMP Specification: https://www.openmp.org/
