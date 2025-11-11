# #!/bin/bash
# set -euo pipefail

# # --- Configuration ---
# BUILD_TYPE="Debug"

# # --- FIX #1: Force the CUDA 13 Compiler via Environment Variable ---
# # This is the most reliable way to tell CMake which compiler to use.
# # Please verify this path is correct for your system.
# export CUDACXX=/usr/local/cuda-13.0/bin/nvcc

# # --- Path Setup ---
# ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# CORE_SRC="$ROOT/cgadimpl"
# CORE_BUILD="$CORE_SRC/build"
# KERNELS_SRC="$ROOT/kernels"
# KERNELS_BUILD="$KERNELS_SRC/build"

# echo "== Build Type:    $BUILD_TYPE"
# echo "== Using CUDA CXX: $CUDACXX"
# echo "== Cleaning build directories for a fresh start..."
# rm -rf "$CORE_BUILD"
# rm -rf "$KERNELS_BUILD"

# # --- Core Build ---
# echo "== Configuring core"
# # The CMAKE_CUDA_COMPILER flag is no longer needed because we exported CUDACXX
# cmake -S "$CORE_SRC" -B "$CORE_BUILD" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

# echo "== Building core"
# cmake --build "$CORE_BUILD" -- -j$(nproc)

# # --- Kernels Build ---
# echo "== Configuring kernel plugins"
# # CUDACXX will be inherited by this command as well
# cmake -S "$KERNELS_SRC" -B "$KERNELS_BUILD" \
#   -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
#   -DCGADIMPL_INCLUDE_DIR="$CORE_SRC/include"

# echo "== Building kernel plugins"
# cmake --build "$KERNELS_BUILD" -- -j$(nproc)

# # --- Staging and Testing ---
# echo "== Copying kernel plugins to test directory"
# cp "$KERNELS_BUILD/cpu/libagkernels_cpu.so" "$CORE_BUILD/"
# cp "$KERNELS_BUILD/gpu/libagkernels_cuda.so" "$CORE_BUILD/"

# echo "== Staging complete. Running tests..."
# cd "$CORE_BUILD"
# ctest --output-on-failure
# cd "$ROOT"

# echo "✅ Build and test run process finished."
#!/bin/bash
set -euo pipefail

# --- Configuration ---
BUILD_TYPE="Debug"

# This is the most reliable way to tell CMake which compiler to use.
export CUDACXX=/usr/local/cuda-13.0/bin/nvcc

# --- Path Setup ---
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CGADIMPL_DIR="$ROOT/cgadimpl"
CGADIMPL_BUILD="$CGADIMPL_DIR/build"
KERNELS_DIR="$ROOT/kernels"
KERNELS_BUILD="$KERNELS_DIR/build"
TENSOR_DIR="$ROOT/tensor"

echo "== Build Type:    $BUILD_TYPE"
echo "== Using CUDA CXX: $(which nvcc)"

# --- Clean all projects for a guaranteed fresh start ---
echo "== Cleaning build directories for a fresh start..."
rm -rf "$CGADIMPL_BUILD"
rm -rf "$KERNELS_BUILD"
# Also clean the old tensor library artifacts to be safe
rm -rf "$TENSOR_DIR/lib" "$TENSOR_DIR/lib/objects"

# =========================================================================
# ====> STEP 1: BUILD THE TENSOR LIBRARY (THIS IS THE FIX) <====
# =========================================================================
echo "== Building tensor library"
cd "${TENSOR_DIR}"
make -j$(nproc)
cd "${ROOT}"

# --- STEP 2: Configure and build the core cgadimpl library ---
echo "== Configuring core"
cmake -S "$CGADIMPL_DIR" -B "$CGADIMPL_BUILD" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

echo "== Building core"
cmake --build "$CGADIMPL_BUILD" -- -j$(nproc)

# --- STEP 3: Configure and build the kernel plugins ---
echo "== Configuring kernel plugins"
cmake -S "$KERNELS_DIR" -B "$KERNELS_BUILD" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCGADIMPL_INCLUDE_DIR="$CGADIMPL_DIR/include"

echo "== Building kernel plugins"
cmake --build "$KERNELS_BUILD" -- -j$(nproc)

# --- STEP 4: Stage build artifacts for testing ---
echo "== Copying kernel plugins to test directory"
cp "$KERNELS_BUILD/cpu/libagkernels_cpu.so" "$CGADIMPL_BUILD/"
cp "$KERNELS_BUILD/gpu/libagkernels_cuda.so" "$CGADIMPL_BUILD/"

# --- STEP 5: Run tests ---
echo "== Staging complete. Running tests..."
cd "$CGADIMPL_BUILD"
ctest --output-on-failure
cd "$ROOT"

echo "✅ Build and test run process finished."