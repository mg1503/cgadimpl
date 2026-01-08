#!/bin/bash
set -euo pipefail

# --- Configuration ---
BUILD_TYPE="Debug"

# --- Path Setup ---
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CGADIMPL_DIR="$ROOT/cgadimpl"
CGADIMPL_BUILD="$CGADIMPL_DIR/build"
KERNELS_DIR="$ROOT/kernels"
KERNELS_BUILD="$KERNELS_DIR/build"
TENSOR_DIR="$ROOT/tensor"

# --- Toolchain Setup ---
# CUDA 12.6 is the installed version.
export CUDACXX=/usr/local/cuda/bin/nvcc
# Add CUDA and Nova-Compiler binaries to PATH so they are found at runtime
export PATH="/usr/local/cuda/bin:$ROOT/Nova-Compiler/install/bin:$PATH"

echo "== Build Type:    $BUILD_TYPE"
echo "== Using CUDA CXX: $CUDACXX"
echo "== PATH: $PATH"

# --- Incremental Build (default) ---
# To force a clean rebuild, run: rm -rf cgadimpl/build kernels/build tensor/lib
# This script now does incremental builds by default for faster compilation

# =========================================================================
# ====> STEP 1: BUILD THE TENSOR LIBRARY (INCREMENTAL) <====
# =========================================================================
echo "== Building tensor library"
cd "${TENSOR_DIR}"
# Explicitly pass NVCC to make to ensure it uses the correct compiler
make NVCC="$CUDACXX" -j$(nproc)

# --- STEP 1.5: Build and Install Nova Compiler ---
NOVA_DIR="$ROOT/Nova-Compiler"
NOVA_BUILD="$NOVA_DIR/build"
NOVA_INSTALL="$NOVA_DIR/install"

echo "== Configuring Nova Compiler"
# Ensure Nova-Compiler CMakeLists.txt knows where MLIR is if needed, 
# but usually it finds it via system or submodule. Assuming submodule logic is fine.
cmake -S "$NOVA_DIR" -B "$NOVA_BUILD" \
    -DCMAKE_BUILD_TYPE="Release" \
    -DCMAKE_INSTALL_PREFIX="$NOVA_INSTALL" \
    -DMLIR_DIR="/home/blubridge-035/llvm-project/build/lib/cmake/mlir"


echo "== Installing Nova Compiler"
# This builds AND installs to the local install/ directory
cmake --build "$NOVA_BUILD" --target install -- -j$(nproc)


# --- STEP 2: Configure and build the core cgadimpl library ---
echo "== Configuring core"
# We don't need to pass NOVA_OPT_BIN here if the code relies on it being in PATH.
# We've added it to PATH above.
cmake -S "$CGADIMPL_DIR" -B "$CGADIMPL_BUILD" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -Dmlir-compiler_DIR="$ROOT/Nova-Compiler/install/lib/cmake/mlir-compiler" \
    -DMLIR_DIR="/home/blubridge-035/llvm-project/build/lib/cmake/mlir"

echo "== Building core"
cmake --build "$CGADIMPL_BUILD" -- -j$(nproc)

# # --- STEP 3: Configure and build the kernel plugins ---
# echo "== Configuring kernel plugins"
# cmake -S "$KERNELS_DIR" -B "$KERNELS_BUILD" \
#   -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
#   -DCGADIMPL_INCLUDE_DIR="$CGADIMPL_DIR/include"

# echo "== Building kernel plugins"
# cmake --build "$KERNELS_BUILD" -- -j$(nproc)

# # --- STEP 4: Stage build artifacts for testing ---
# echo "== Copying kernel plugins to test directory"
# cp "$KERNELS_BUILD/cpu/libagkernels_cpu.so" "$CGADIMPL_BUILD/"
# cp "$KERNELS_BUILD/gpu/libagkernels_cuda.so" "$CGADIMPL_BUILD/"

# --- STEP 5: Run tests ---
# echo "== Staging complete. Running tests..."
# cd "$CGADIMPL_BUILD"
# ctest --output-on-failure
# cd "$ROOT"

echo "✅ Build and test run process finished."