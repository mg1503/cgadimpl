
cmake_minimum_required(VERSION 3.20)

# --- FIX #2: Set the CUDA architecture BEFORE the project() command ---
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA SMs" FORCE)

project(cgadimpl LANGUAGES CXX CUDA)

# --- Compiler, CUDA, and OpenMP Setup ---
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Find and Link the Pre-built Tensor Library ---
set(OWNTENSOR_DIR ${CMAKE_SOURCE_DIR}/../tensor)
find_path(OWNTENSOR_INCLUDE_DIR NAMES TensorLib.h HINTS ${OWNTENSOR_DIR}/include)
find_library(OWNTENSOR_LIBRARY NAMES tensor HINTS ${OWNTENSOR_DIR}/lib)

if(OWNTENSOR_INCLUDE_DIR AND OWNTENSOR_LIBRARY)
    add_library(OwnTensor::tensor SHARED IMPORTED)
    set_target_properties(OwnTensor::tensor PROPERTIES
        IMPORTED_LOCATION "${OWNTENSOR_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OWNTENSOR_INCLUDE_DIR}"
    )
    message(STATUS "Found OwnTensor headers: ${OWNTENSOR_INCLUDE_DIR}")
    message(STATUS "Found OwnTensor library: ${OWNTENSOR_LIBRARY}")
else()
    message(FATAL_ERROR "Could not find the OwnTensor library. Searched in ${OWNTENSOR_DIR}")
endif()

# ---- Sources ----
file(GLOB_RECURSE CGADIMPL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# ---- Library Definition ----
add_library(cgadimpl STATIC ${CGADIMPL_SRC})
add_library(cgadimpl::cgadimpl ALIAS cgadimpl)

target_compile_definitions(cgadimpl PUBLIC WITH_CUDA)
target_include_directories(cgadimpl PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(cgadimpl PUBLIC OwnTensor::tensor dl)

# ---- Tests ----
if(true)
  enable_testing()
  function(add_ag_test name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE cgadimpl::cgadimpl CUDA::cudart OpenMP::OpenMP_CXX)
    add_test(NAME ${name} COMMAND ${name})
  endfunction()
  
  set(CUDA_TEST_SOURCES tests/test_kernels_gpu.cpp tests/test_graph_gpu.cpp tests/test_end_to_end_gpu.cpp)
  set_source_files_properties(${CUDA_TEST_SOURCES} PROPERTIES LANGUAGE CUDA)

  add_ag_test(test_ag                    tests/test_ag.cpp)
  add_ag_test(test_mlp                    tests/test_mlp.cpp)
  add_ag_test(test_complex_mlp                    tests/test_complex_mlp.cpp)
  add_ag_test(test_bench_relu                    tests/bench_relu.cpp)
  add_ag_test(test_tiny_handcalc                    tests/tiny_handcalc.cpp)
  add_ag_test(test_graph_compile                    tests/test_graph_compile.cpp)
  add_ag_test(ag_core_test                    tests/test.cpp)
  add_ag_test(test_checkpoint                    tests/test_checkpoint.cpp)
  add_ag_test(test_inplace                    tests/test_inplace.cpp)
  add_ag_test(test_careful_deletion                    tests/test_careful_deletion.cpp)
  add_ag_test(test_nn                    tests/test_nn.cpp)
  add_ag_test(test_version                    tests/test_version.cpp)
  add_ag_test(test_nodeops                    tests/test_nodeops.cpp)
  add_ag_test(test_kernels_cpu                    tests/test_kernels_cpu.cpp)
  add_ag_test(test_vjp_kernels                    tests/test_end_to_end_gpu.cpp)
  add_ag_test(test_nn_mlp                    tests/test_mlp_training.cpp)
  add_ag_test(test_kernels_gpu                    tests/test_kernels_gpu.cpp)
  add_ag_test(test_graph_cuda                    tests/test_graph_gpu.cpp)
  add_ag_test(test_core_ops_gpu                    tests/test_core_ops_gpu.cpp)
  add_ag_test(test_kitchen_sink_mlp                   tests/test_kitchen_sink_mlp.cpp)
  add_ag_test(test_gantlet                    tests/test_gantlet.cpp)
endif()

message(STATUS "cgadimpl build mode: ${CMAKE_BUILD_TYPE}")