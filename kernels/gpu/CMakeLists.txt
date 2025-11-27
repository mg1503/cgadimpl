# # #  ================================
# # #  kernels/gpu/CMakeLists.txt
# # #  ================================
# # cmake_minimum_required(VERSION 3.20)
# # project(agkernels_cuda LANGUAGES CXX CUDA)

# # set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "Target SM architecture" FORCE)
# # # --- Force modern C++ on both host and device ---
# # set(CMAKE_CXX_STANDARD 20)
# # set(CMAKE_CXX_STANDARD_REQUIRED ON)
# # set(CMAKE_CUDA_STANDARD 20)                 # 11.5 supports C++17
# # set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# # add_library(agkernels_cuda SHARED
# #   entry.cu
# #   eltwise.cu
# #   mm_cublas.cu
# #   zero.cu
# #   vjp.cu 
# #   # matker.cu # <-- TO BE ADDED LATER
# # )

# # set_target_properties(agkernels_cuda PROPERTIES
# #   CUDA_SEPARABLE_COMPILATION ON
# #   POSITION_INDEPENDENT_CODE ON
# #   OUTPUT_NAME agkernels_cuda
# #   # Optional but recommended – pick a reasonable default arch if none is set:
# #   # CUDA_ARCHITECTURES "60;70;75;80;86;89"
# #     CUDA_ARCHITECTURES "86" # for testing on rtx 30 series
# #   )

# # find_package(CUDAToolkit REQUIRED)
# # target_link_libraries(agkernels_cuda PUBLIC
# #   CUDA::cudart
# #   CUDA::cublas
# #   CUDA::cublasLt
# # )

# # # Header path to ad/kernels_api.hpp
# # target_include_directories(agkernels_cuda PUBLIC
# #   "${CGADIMPL_INCLUDE_DIR}"
# # )

# # # --- Belt & suspenders: explicitly pass standards to nvcc and host compiler ---
# # target_compile_features(agkernels_cuda PUBLIC cxx_std_20)
# # target_compile_options(agkernels_cuda PRIVATE
# #   $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>
# #   $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++17>
# # )

# # # If your distro mixes older libstdc++/gcc, this can sometimes be required:
# # # target_compile_definitions(agkernels_cuda PRIVATE _GLIBCXX_USE_CXX11_ABI=1)


# cmake_minimum_required(VERSION 3.20)

# # This must come BEFORE the project() command.
# set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "Target SM architecture" FORCE)

# project(agkernels_cuda LANGUAGES CXX CUDA)

# # --- FIX: Use the robust "Bait and Switch" for C++20 Standard ---
# # This matches your main cgadimpl project and bypasses the CMake knowledge gap.
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# # Tell CMake we want C++17 for its internal checks...
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# # ...then manually force the C++20 flag for the compiler, which overrides the C++17 one.
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")
# # --- END FIX ---

# add_library(agkernels_cuda SHARED
#   entry.cu
#   eltwise.cu
#   mm_cublas.cu
#   zero.cu
#   vjp.cu 
# )

# set_target_properties(agkernels_cuda PROPERTIES
#   CUDA_SEPARABLE_COMPILATION ON
#   POSITION_INDEPENDENT_CODE ON
#   OUTPUT_NAME agkernels_cuda
# )

# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(agkernels_cuda PUBLIC
#   CUDA::cudart
#   CUDA::cublas
#   CUDA::cublasLt
# )

# # This is correct
# target_include_directories(agkernels_cuda PUBLIC
#   "${CGADIMPL_INCLUDE_DIR}"
# )
cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "Target SM architecture" FORCE)
project(agkernels_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")

add_library(agkernels_cuda SHARED entry.cu eltwise.cu mm_cublas.cu zero.cu vjp.cu)
set_target_properties(agkernels_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON OUTPUT_NAME agkernels_cuda)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(agkernels_cuda PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt)
target_include_directories(agkernels_cuda PUBLIC "${CGADIMPL_INCLUDE_DIR}")