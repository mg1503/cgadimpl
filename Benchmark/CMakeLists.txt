cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mlp_manual)

# 1. Force RTX 3090 architecture (8.6) and BLOCK sm_52 verification
set(TORCH_CUDA_ARCH_LIST "8.6" CACHE STRING "CUDA architectures" FORCE)
set(CMAKE_CUDA_ARCHITECTURES 86)

# 2. Point to the correct CUDA 13 path to avoid /usr/bin/nvcc stub errors
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# 3. LibTorch path
set(CMAKE_PREFIX_PATH "/home/ble-bridge022/Desktop/grishma/cgadimpl/libtorch")
find_package(Torch REQUIRED)

add_executable(mlp_manual test.cpp)
target_link_libraries(mlp_manual "${TORCH_LIBRARIES}")
set_property(TARGET mlp_manual PROPERTY CXX_STANDARD 17)
