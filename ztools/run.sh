#!/usr/bin/env bash
# Build cgadimpl (core) in ./cgadimpl and the kernels/cpu plugin in ./kernels.
# Usage:
#   bash ztools/run.sh [--type Release|Debug] [--clean]
# Optional:
#   bash ztools/run.sh --type Debug
#   bash ztools/run.sh --clean

set -euo pipefail

BUILD_TYPE="Release"
CLEAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --type)  BUILD_TYPE="${2:-Release}"; shift 2;;
    --clean) CLEAN=1; shift;;
    -h|--help) grep -m1 -A5 '^# Build cgadimpl' "$0"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# Repo root = parent of this script
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORE_SRC="$ROOT/cgadimpl"
CORE_BUILD="$CORE_SRC/build"
CORE_INCLUDE="$CORE_SRC/include"   # contains include/ad/kernels_api.hpp
KERNELS_SRC="$ROOT/kernels"
KERNELS_BUILD="$KERNELS_SRC/build"

if [[ ! -d "$CORE_INCLUDE/ad" ]]; then
  echo "Expected headers at: $CORE_INCLUDE/ad"; exit 1
fi
if [[ ! -d "$KERNELS_SRC/cpu/src" ]]; then
  echo "Expected kernels at: $KERNELS_SRC/cpu/src"; exit 1
fi

# OS shared-lib suffix
case "$(uname -s)" in
  Linux*)  SO_SUFFIX="so";   LIBVAR="LD_LIBRARY_PATH";;
  Darwin*) SO_SUFFIX="dylib"; LIBVAR="DYLD_LIBRARY_PATH";;
  *) echo "Unsupported OS"; exit 1;;
esac

echo "== Root:          $ROOT"
echo "== Core (src):    $CORE_SRC"
echo "== Kernels (src): $KERNELS_SRC"
echo "== Type:          $BUILD_TYPE"

if [[ $CLEAN -eq 1 ]]; then
  echo "== Cleaning build dirs"
  rm -rf "$CORE_BUILD" "$KERNELS_BUILD"
fi

echo "== Configuring core"
cmake -S "$CORE_SRC" -B "$CORE_BUILD" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

echo "== Building core"
cmake --build "$CORE_BUILD" -j

echo "== Configuring kernels/cpu"
cmake -S "$KERNELS_SRC" -B "$KERNELS_BUILD" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCGADIMPL_INCLUDE_DIR="$CORE_INCLUDE"

echo "== Building kernels/cpu"
cmake --build "$KERNELS_BUILD" -j

cmake -S "$CORE_SRC" -B "$CORE_BUILD" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DAG_BUILD_TESTS=ON     # or OFF if you prefer


# Locate plugin
PLUGIN_CANDIDATES=(
  "$KERNELS_BUILD/cpu/libagkernels_cpu.${SO_SUFFIX}"
  "$KERNELS_BUILD/cpu/agkernels_cpu.${SO_SUFFIX}"
  "$KERNELS_BUILD/libagkernels_cpu.${SO_SUFFIX}"
  "$KERNELS_BUILD/agkernels_cpu.${SO_SUFFIX}"
  "$KERNELS_BUILD/cpu/${BUILD_TYPE}/agkernels_cpu.${SO_SUFFIX}"
)
PLUGIN_PATH=""
for p in "${PLUGIN_CANDIDATES[@]}"; do
  [[ -f "$p" ]] && { PLUGIN_PATH="$p"; break; }
done
[[ -n "$PLUGIN_PATH" ]] || { echo "!! Could not find built plugin"; printf '   looked: %s\n' "${PLUGIN_CANDIDATES[@]}"; exit 1; }

STAGED_PLUGIN="$CORE_BUILD/$(basename "$PLUGIN_PATH")"
cp -f "$PLUGIN_PATH" "$STAGED_PLUGIN"

cat <<EOF

Build complete.

Core build dir:
  $CORE_BUILD

CPU plugin staged next to it:
  $STAGED_PLUGIN

Run from \$CORE_BUILD and load the plugin like:
  ag::kernels::load_cpu_plugin("./$(basename "$STAGED_PLUGIN")");

If running elsewhere, ensure the loader can find it:
  export ${LIBVAR}=\$${LIBVAR}:$(dirname "$STAGED_PLUGIN")

EOF
