# # # # ====================
# # # # ztools/run.sh
# # # # ====================

# # # #!/usr/bin/env bash
# # # # Build cgadimpl (core) in ./cgadimpl and the kernels/cpu plugin in ./kernels.
# # # # Usage:
# # # #   bash ztools/run.sh [--type Release|Debug] [--clean]
# # # # Optional:
# # # #   bash ztools/run.sh --type Debug
# # # #   bash ztools/run.sh --clean

# # # set -euo pipefail

# # # BUILD_TYPE="Release"
# # # CLEAN=0

# # # while [[ $# -gt 0 ]]; do
# # #   case "$1" in
# # #     --type)  BUILD_TYPE="${2:-Release}"; shift 2;;
# # #     --clean) CLEAN=1; shift;;
# # #     -h|--help) grep -m1 -A5 '^# Build cgadimpl' "$0"; exit 0;;
# # #     *) echo "Unknown arg: $1"; exit 1;;
# # #   esac
# # # done

# # # # Repo root = parent of this script
# # # ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# # # CORE_SRC="$ROOT/cgadimpl"
# # # CORE_BUILD="$CORE_SRC/build"
# # # CORE_INCLUDE="$CORE_SRC/include"
# # # KERNELS_SRC="$ROOT/kernels"
# # # KERNELS_BUILD="$KERNELS_SRC/build"

# # # if [[ ! -d "$CORE_INCLUDE/ad" ]]; then
# # #   echo "Expected headers at: $CORE_INCLUDE/ad"; exit 1
# # # fi
# # # if [[ ! -d "$KERNELS_SRC/cpu/src" ]]; then
# # #   echo "Expected kernels at: $KERNELS_SRC/cpu/src"; exit 1
# # # fi

# # # # OS shared-lib suffix
# # # case "$(uname -s)" in
# # #   Linux*)  SO_SUFFIX="so";   LIBVAR="LD_LIBRARY_PATH";;
# # #   Darwin*) SO_SUFFIX="dylib"; LIBVAR="DYLD_LIBRARY_PATH";;
# # #   *) echo "Unsupported OS"; exit 1;;
# # # esac

# # # echo "== Root:          $ROOT"
# # # echo "== Core (src):    $CORE_SRC"
# # # echo "== Kernels (src): $KERNELS_SRC"
# # # echo "== Type:          $BUILD_TYPE"

# # # if [[ $CLEAN -eq 1 ]]; then
# # #   echo "== Cleaning build dirs"
# # #   rm -rf "$CORE_BUILD" "$KERNELS_BUILD"
# # # fi

# # # echo "== Configuring core"
# # # # --- FIX: Add -DCMAKE_CUDA_COMPILER to force CUDA 13 ---
# # # cmake -S "$CORE_SRC" -B "$CORE_BUILD" \
# # #   -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
# # #   -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc

# # # echo "== Building core"
# # # cmake --build "$CORE_BUILD" -- -j$(nproc)

# # # echo "== Configuring kernels/cpu"
# # # # --- FIX: Add -DCMAKE_CUDA_COMPILER here as well ---
# # # cmake -S "$KERNELS_SRC" -B "$KERNELS_BUILD" \
# # #   -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
# # #   -DCGADIMPL_INCLUDE_DIR="$CORE_INCLUDE" \
# # #   -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc

# # # echo "== Building kernels"
# # # cmake --build "$KERNELS_BUILD" -- -j$(nproc)

# # # # ... (rest of script is unchanged) ...

# # # # Locate plugin
# # # PLUGIN_CANDIDATES=(

# # #   "$KERNELS_BUILD/cpu/libagkernels_cpu.${SO_SUFFIX}"
# # #   "$KERNELS_BUILD/cpu/agkernels_cpu.${SO_SUFFIX}"
# # #   "$KERNELS_BUILD/libagkernels_cpu.${SO_SUFFIX}"
# # #   "$KERNELS_BUILD/agkernels_cpu.${SO_SUFFIX}"
# # #   "$KERNELS_BUILD/cpu/${BUILD_TYPE}/agkernels_cpu.${SO_SUFFIX}"
# # # )
# # # PLUGIN_PATH=""
# # # for p in "${PLUGIN_CANDIDATES[@]}"; do
# # #   [[ -f "$p" ]] && { PLUGIN_PATH="$p"; break; }
# # # done
# # # [[ -n "$PLUGIN_PATH" ]] || { echo "!! Could not find built plugin"; printf '   looked: %s\n' "${PLUGIN_CANDIDATES[@]}"; exit 1; }

# # # STAGED_PLUGIN="$CORE_BUILD/$(basename "$PLUGIN_PATH")"
# # # cp -f "$PLUGIN_PATH" "$STAGED_PLUGIN"

# # # # Stage CUDA plugin if present
# # # CUDA_CANDIDATES=(
# # #   "$KERNELS_BUILD/gpu/libagkernels_cuda.${SO_SUFFIX}"
# # #   "$KERNELS_BUILD/gpu/${BUILD_TYPE}/libagkernels_cuda.${SO_SUFFIX}"
# # #   "$KERNELS_BUILD/gpu/agkernels_cuda.${SO_SUFFIX}"
# # #   "$KERNELS_BUILD/libagkernels_cuda.${SO_SUFFIX}"
# # # )


# # # CUDA_PLUGIN=""
# # # for p in "${CUDA_CANDIDATES[@]}"; do
# # #   if [[ -f "$p" ]]; then CUDA_PLUGIN="$p"; break; fi
# # # done

# # # if [[ -n "$CUDA_PLUGIN" ]]; then
# # #   cp -f "$CUDA_PLUGIN" "$CORE_BUILD/"
# # #   echo "Staged CUDA plugin: $CORE_BUILD/$(basename "$CUDA_PLUGIN")"
# # # else
# # #   echo "CUDA plugin not found — skipping stage (CPU-only ok)."
# # # fi
# # # cat <<EOF

# # # Build complete.

# # # Core build dir:
# # #   $CORE_BUILD

# # # CPU plugin staged next to it:
# # #   $STAGED_PLUGIN

# # # Run from \$CORE_BUILD and load the plugin like:
# # #   ag::kernels::load_cpu_plugin("./$(basename "$STAGED_PLUGIN")");

# # # If running elsewhere, ensure the loader can find it:
# # #   export ${LIBVAR}=\$${LIBVAR}:$(dirname "$STAGED_PLUGIN")

# # # EOF
# # #!/usr/bin/env bash
# # # Builds cgadimpl (core) and the kernel plugins, then runs tests.

# # set -euo pipefail

# # BUILD_TYPE="Debug" 
# # CLEAN=0 
# # # --- Configuration ---
# # BUILD_TYPE="Release"
# # CLEAN=0 # Set to 1 to force clean build

# # # --- Path Setup ---
# # ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# # CORE_SRC="$ROOT/cgadimpl"
# # CORE_BUILD="$CORE_SRC/build"
# # KERNELS_SRC="$ROOT/kernels"
# # KERNELS_BUILD="$KERNELS_SRC/build"

# # echo "== Root:          $ROOT"
# # echo "== Core (src):    $CORE_SRC"
# # echo "== Kernels (src): $KERNELS_SRC"
# # echo "== Type:          $BUILD_TYPE"

# # # --- Core Build ---
# # if [[ $CLEAN -eq 1 ]]; then
# #   echo "== Cleaning core build directory"
# #   rm -rf "$CORE_BUILD"
# # fi

# # echo "== Configuring core"
# # cmake -S "$CORE_SRC" -B "$CORE_BUILD" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

# # echo "== Building core"
# # cmake --build "$CORE_BUILD" -- -j$(nproc)

# # # --- Kernels Build (Clean configure every time to be safe) ---
# # echo "== Cleaning and configuring kernel plugins"
# # rm -rf "$KERNELS_BUILD" # Always clean kernels to avoid cache issues

# # cmake -S "$KERNELS_SRC" -B "$KERNELS_BUILD" \
# #   -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
# #   -DCGADIMPL_INCLUDE_DIR="$CORE_SRC/include" \
# #   -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc

# # echo "== Building kernel plugins"
# # cmake --build "$KERNELS_BUILD" -- -j$(nproc)

# # # --- Staging and Testing ---
# # echo "== Copying kernel plugins to test directory"
# # cp "$KERNELS_BUILD/cpu/libagkernels_cpu.so" "$CORE_BUILD/"
# # cp "$KERNELS_BUILD/gpu/libagkernels_cuda.so" "$CORE_BUILD/"

# # echo "== Running tests from cgadimpl/build"
# # cd "$CORE_BUILD"
# # ctest --output-on-failure
# # cd "$ROOT"

# # echo "✅ Build and test run completed successfully."
# #!/usr/bin/env bash
# # Builds the entire project in Debug mode, copies plugins, and runs tests.

# set -euo pipefail

# # --- Configuration ---
# BUILD_TYPE="Debug"

# # --- Path Setup ---
# ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# CORE_SRC="$ROOT/cgadimpl"
# CORE_BUILD="$CORE_SRC/build"
# KERNELS_SRC="$ROOT/kernels"
# KERNELS_BUILD="$KERNELS_SRC/build"

# echo "== Build Type:    $BUILD_TYPE"
# echo "== Cleaning build directories for a fresh start..."
# rm -rf "$CORE_BUILD"
# rm -rf "$KERNELS_BUILD"

# # --- Core Build ---
# echo "== Configuring core"
# cmake -S "$CORE_SRC" -B "$CORE_BUILD" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

# echo "== Building core"
# cmake --build "$CORE_BUILD" -- -j$(nproc)

# # --- Kernels Build ---
# echo "== Configuring kernel plugins"
# cmake -S "$KERNELS_SRC" -B "$KERNELS_BUILD" \
#   -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
#   -DCGADIMPL_INCLUDE_DIR="$CORE_SRC/include" \
#   -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc

# echo "== Building kernel plugins"
# cmake --build "$KERNELS_BUILD" -- -j$(nproc)

# # --- Staging and Testing ---
# echo "== Copying kernel plugins to test directory"
# cp "$KERNELS_BUILD/cpu/libagkernels_cpu.so" "$CORE_BUILD/"
# cp "$KERNELS_BUILD/gpu/libagkernels_cuda.so" "$CORE_BUILD/"

# echo "== Running tests from cgadimpl/build"
# cd "$CORE_BUILD"
# ctest --output-on-failure
# cd "$ROOT"

# echo "✅ Build and test run completed."
#!/usr/bin/env bash
# Builds the entire project in Debug mode, copies plugins, and runs tests.

set -euo pipefail

# --- Configuration ---
BUILD_TYPE="Debug"

# --- FIX #1: Force the CUDA 13 Compiler via Environment Variable ---
# This is the most reliable way to tell CMake which compiler to use.
# Please verify this path is correct for your system.
export CUDACXX=/usr/local/cuda-13.0/bin/nvcc

# --- Path Setup ---
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORE_SRC="$ROOT/cgadimpl"
CORE_BUILD="$CORE_SRC/build"
KERNELS_SRC="$ROOT/kernels"
KERNELS_BUILD="$KERNELS_SRC/build"

echo "== Build Type:    $BUILD_TYPE"
echo "== Using CUDA CXX: $CUDACXX"
echo "== Cleaning build directories for a fresh start..."
rm -rf "$CORE_BUILD"
rm -rf "$KERNELS_BUILD"

# --- Core Build ---
echo "== Configuring core"
# The CMAKE_CUDA_COMPILER flag is no longer needed because we exported CUDACXX
cmake -S "$CORE_SRC" -B "$CORE_BUILD" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

echo "== Building core"
cmake --build "$CORE_BUILD" -- -j$(nproc)

# --- Kernels Build ---
echo "== Configuring kernel plugins"
# CUDACXX will be inherited by this command as well
cmake -S "$KERNELS_SRC" -B "$KERNELS_BUILD" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCGADIMPL_INCLUDE_DIR="$CORE_SRC/include"

echo "== Building kernel plugins"
cmake --build "$KERNELS_BUILD" -- -j$(nproc)

# --- Staging and Testing ---
echo "== Copying kernel plugins to test directory"
cp "$KERNELS_BUILD/cpu/libagkernels_cpu.so" "$CORE_BUILD/"
cp "$KERNELS_BUILD/gpu/libagkernels_cuda.so" "$CORE_BUILD/"

echo "== Staging complete. Running tests..."
cd "$CORE_BUILD"
ctest --output-on-failure
cd "$ROOT"

echo "✅ Build and test run process finished."