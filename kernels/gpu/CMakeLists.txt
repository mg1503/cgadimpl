#  ================================
#  kernels/gpu/CMakeLists.txt
#  ================================
cmake_minimum_required(VERSION 3.20)
project(agkernels_cuda LANGUAGES CXX CUDA)

# --- Force modern C++ on both host and device ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)                 # 11.5 supports C++17
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(agkernels_cuda SHARED
  entry.cu
  eltwise.cu
  mm_cublas.cu
  zero.cu
  vjp.cu 
)

set_target_properties(agkernels_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME agkernels_cuda
  # Optional but recommended – pick a reasonable default arch if none is set:
  # CUDA_ARCHITECTURES "60;70;75;80;86;89"
    CUDA_ARCHITECTURES "86" # for testing on rtx 30 series
  )

find_package(CUDAToolkit REQUIRED)
target_link_libraries(agkernels_cuda PUBLIC
  CUDA::cudart
  CUDA::cublas
  CUDA::cublasLt
)

# Header path to ad/kernels_api.hpp
target_include_directories(agkernels_cuda PUBLIC
  "${CGADIMPL_INCLUDE_DIR}"
)

# --- Belt & suspenders: explicitly pass standards to nvcc and host compiler ---
target_compile_features(agkernels_cuda PUBLIC cxx_std_17)
target_compile_options(agkernels_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>
  $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++17>
)

# If your distro mixes older libstdc++/gcc, this can sometimes be required:
# target_compile_definitions(agkernels_cuda PRIVATE _GLIBCXX_USE_CXX11_ABI=1)


# //other approach
# cmake_minimum_required(VERSION 3.20)
# project(agkernels_cuda LANGUAGES CXX CUDA)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# add_library(agkernels_cuda SHARED
#   entry.cu
#   eltwise.cu
#   zero.cu
#   custom_kernels.cu  # <-- ADDED
# )

# set_target_properties(agkernels_cuda PROPERTIES
#   CUDA_SEPARABLE_COMPILATION ON
#   POSITION_INDEPENDENT_CODE ON
#   OUTPUT_NAME agkernels_cuda
#   CUDA_ARCHITECTURES "60;70;75;80"
# )

# # We no longer need cuBLAS! But we still need the runtime.
# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(agkernels_cuda PUBLIC CUDA::cudart)


# target_include_directories(agkernels_cuda PUBLIC
#   "${CGADIMPL_INCLUDE_DIR}"
# )

# target_compile_features(agkernels_cuda PUBLIC cxx_std_17)
# target_compile_options(agkernels_cuda PRIVATE
#   $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>
#   $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++17>
# )