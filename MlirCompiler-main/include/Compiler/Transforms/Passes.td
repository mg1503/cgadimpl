#ifndef NOVA_TRANSFORMS_PASSES
#define NOVA_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"


def AffineFullUnroll:Pass<"affine-full-unroll">{
    let summary="Fully unroll all affine loops";
    let description=[{
        Full unroll all affine loops
    }];
    let dependentDialects=["mlir::affine::AffineDialect"];
}

def FastmathFlag : Pass<"fastmath-flag", "func::FuncOp"> {
    let summary = "Add fastmath<contract> flags to enable FMA fusion";
    let description = [{
        Insert a fastmath flag so that math-uplift-to-fma pass can be triggered
    }];
    let dependentDialects = ["mlir::arith::ArithDialect"];
    let constructor = "::mlir::nova::createFastmathFlagPass()";
}

def ParallelizeOuterLoops : Pass<"parallelize-outer-loops", "func::FuncOp"> {
    let summary = "Mark outer matmul loops (i,j) as parallel while preserving inner loops for vectorization";
    let description = [{
        Converts the two outermost affine.for loops to scf.parallel,
        enabling thread-level parallelism via OpenMP while leaving
        inner loops (including reduction) intact for vectorization.

        This avoids the conflict between affine-parallelize and
        affine-super-vectorize by operating at the right granularity.
    }];
    let dependentDialects = [
        "mlir::affine::AffineDialect",
        "mlir::scf::SCFDialect"
    ];
    let constructor = "::mlir::nova::createParallelizeOuterLoopsPass()";
}

def FuseMatmulBias : Pass<"fuse-matmul-bias", "func::FuncOp"> {
    let summary = "Fuse matmul + bias add operations";
    let description = [{
        Detects the pattern of matmul followed by elementwise add (bias)
        and fuses them into a single operation to eliminate intermediate
        memory traffic. This is a common pattern in neural networks.
    }];
    let dependentDialects = [
        "mlir::linalg::LinalgDialect",
        "mlir::arith::ArithDialect"
    ];
    let constructor = "::mlir::nova::createFuseMatmulBiasPass()";
}

#endif // NOVA_TRANSFORMS_PASSES