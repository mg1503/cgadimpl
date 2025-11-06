# ====================================================================
# FILE: cgadimpl/CMakeLists.txt (Definitive Corrected Version)
# ====================================================================
cmake_minimum_required(VERSION 3.20)
project(cgadimpl LANGUAGES CXX CUDA)

# --- Compiler and CUDA Setup ---
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA SMs" FORCE)
find_package(CUDAToolkit REQUIRED)

# --- Find and Link the Pre-built Tensor Library ---
# 1. Define the path to the 'tensor' library directory.
set(OWNTENSOR_DIR ${CMAKE_SOURCE_DIR}/../tensor)
message(STATUS "Found OwnTensor library at: ${OWNTENSOR_DIR}")

# 2. Tell the linker where to find the compiled library file (libtensor.so).
link_directories(${OWNTENSOR_DIR}/lib)

if(NOT OWNTENSOR_INCLUDE_DIR OR NOT OWNTENSOR_LIBRARY)
    message(FATAL_ERROR "Could not find the OwnTensor library. Searched in ${CMAKE_CURRENT_SOURCE_DIR}/../tensor")
endif()

message(STATUS "Found OwnTensor headers: ${OWNTENSOR_INCLUDE_DIR}")
message(STATUS "Found OwnTensor library: ${OWNTENSOR_LIBRARY}")

# --- Project Options ---
option(AG_GLOB_SOURCES "Glob all .cpp under src/ (simplifies dev)" ON)
option(AG_BUILD_TESTS  "Build tests in tests/" ON)

# --- 1. RESTORED C++ STANDARD "BAIT AND SWITCH" ---
# This is the original, robust method for setting the C++ standard
# that works on your system.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Tell CMake we want C++17 for CUDA to satisfy its internal checks...
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# ...then manually force the C++20 flag, which overrides the C++17 one.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")
# ---------------------------------------------------

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Sources ----
if(AG_GLOB_SOURCES)
  file(GLOB_RECURSE CGADIMPL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
endif()

# ---- Library Definition ----
add_library(cgadimpl STATIC ${CGADIMPL_SRC})
add_library(cgadimpl::cgadimpl ALIAS cgadimpl)

# --- Define Include Directories ---
target_include_directories(cgadimpl PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${OWNTENSOR_INCLUDE_DIR}>
)

# --- Define Library Links ---
target_link_libraries(cgadimpl PUBLIC tensor dl )

# ---- Tests (optional) ----
if(AG_BUILD_TESTS)
  enable_testing()

  function(add_ag_test name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE cgadimpl::cgadimpl CUDA::cudart)
    add_test(NAME ${name} COMMAND ${name})
  endfunction()
  
  set(CUDA_TEST_SOURCES
      tests/test_kernels_gpu.cpp
      tests/test_graph_gpu.cpp
      tests/test_end_to_end_gpu.cpp
  )
  set_source_files_properties(${CUDA_TEST_SOURCES} PROPERTIES LANGUAGE CUDA)

  # Add all tests
  add_ag_test(test_ag                tests/test_ag.cpp)
  add_ag_test(test_mlp               tests/test_mlp.cpp)
  add_ag_test(test_complex_mlp       tests/test_complex_mlp.cpp)
  add_ag_test(test_bench_relu        tests/bench_relu.cpp)
  add_ag_test(test_tiny_handcalc     tests/tiny_handcalc.cpp)
  add_ag_test(test_graph_compile     tests/test_graph_compile.cpp)
  add_ag_test(ag_core_test           tests/test.cpp)
  add_ag_test(test_checkpoint        tests/test_checkpoint.cpp)
  add_ag_test(test_inplace           tests/test_inplace.cpp)
  add_ag_test(test_careful_deletion  tests/test_careful_deletion.cpp)
  add_ag_test(test_nn                tests/test_nn.cpp)
  add_ag_test(test_version           tests/test_version.cpp)
  add_ag_test(test_nodeops           tests/test_nodeops.cpp)
  add_ag_test(test_kernels_cpu       tests/test_kernels_cpu.cpp)
  add_ag_test(test_vjp_kernels       tests/test_end_to_end_gpu.cpp)
  add_ag_test(test_nn_mlp            tests/test_mlp_training.cpp)
  add_ag_test(test_kernels_gpu       tests/test_kernels_gpu.cpp)
  add_ag_test(test_graph_cuda        tests/test_graph_gpu.cpp)
endif()

message(STATUS "cgadimpl build mode: ${CMAKE_BUILD_TYPE}")