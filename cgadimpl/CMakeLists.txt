# # # # # ====================================================================
# # # # # FILE: cgadimpl/CMakeLists.txt (Definitive Corrected Version)
# # # # # ====================================================================
# # # # cmake_minimum_required(VERSION 3.20)
# # # # project(cgadimpl LANGUAGES CXX CUDA)

# # # # # --- Compiler and CUDA Setup ---
# # # # set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA SMs" FORCE)
# # # # find_package(CUDAToolkit REQUIRED)

# # # # # --- Find and Link the Pre-built Tensor Library ---
# # # # # 1. Define the path to the 'tensor' library directory.
# # # # set(OWNTENSOR_DIR ${CMAKE_SOURCE_DIR}/../tensor)
# # # # message(STATUS "Found OwnTensor library at: ${OWNTENSOR_DIR}")

# # # # # 2. Tell the linker where to find the compiled library file (libtensor.so).
# # # # link_directories(${OWNTENSOR_DIR}/lib)

# # # # if(NOT OWNTENSOR_INCLUDE_DIR OR NOT OWNTENSOR_LIBRARY)
# # # #     message(FATAL_ERROR "Could not find the OwnTensor library. Searched in ${CMAKE_CURRENT_SOURCE_DIR}/../tensor")
# # # # endif()

# # # # message(STATUS "Found OwnTensor headers: ${OWNTENSOR_INCLUDE_DIR}")
# # # # message(STATUS "Found OwnTensor library: ${OWNTENSOR_LIBRARY}")

# # # # # --- Project Options ---
# # # # option(AG_GLOB_SOURCES "Glob all .cpp under src/ (simplifies dev)" ON)
# # # # option(AG_BUILD_TESTS  "Build tests in tests/" ON)

# # # # # --- 1. RESTORED C++ STANDARD "BAIT AND SWITCH" ---
# # # # # This is the original, robust method for setting the C++ standard
# # # # # that works on your system.
# # # # set(CMAKE_CXX_STANDARD 20)
# # # # set(CMAKE_CXX_STANDARD_REQUIRED ON)
# # # # # Tell CMake we want C++17 for CUDA to satisfy its internal checks...
# # # # set(CMAKE_CUDA_STANDARD 17)
# # # # set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# # # # # ...then manually force the C++20 flag, which overrides the C++17 one.
# # # # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")
# # # # # ---------------------------------------------------

# # # # set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# # # # # ---- Sources ----
# # # # if(AG_GLOB_SOURCES)
# # # #   file(GLOB_RECURSE CGADIMPL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
# # # # endif()

# # # # # ---- Library Definition ----
# # # # add_library(cgadimpl STATIC ${CGADIMPL_SRC})
# # # # add_library(cgadimpl::cgadimpl ALIAS cgadimpl)

# # # # # --- Define Include Directories ---
# # # # target_include_directories(cgadimpl PUBLIC
# # # #   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
# # # #   $<BUILD_INTERFACE:${OWNTENSOR_INCLUDE_DIR}>
# # # # )

# # # # # --- Define Library Links ---
# # # # target_link_libraries(cgadimpl PUBLIC tensor dl )

# # # # # ---- Tests (optional) ----
# # # # if(AG_BUILD_TESTS)
# # # #   enable_testing()

# # # #   function(add_ag_test name src)
# # # #     add_executable(${name} ${src})
# # # #     target_link_libraries(${name} PRIVATE cgadimpl::cgadimpl CUDA::cudart)
# # # #     add_test(NAME ${name} COMMAND ${name})
# # # #   endfunction()
  
# # # #   set(CUDA_TEST_SOURCES
# # # #       tests/test_kernels_gpu.cpp
# # # #       tests/test_graph_gpu.cpp
# # # #       tests/test_end_to_end_gpu.cpp
# # # #   )
# # # #   set_source_files_properties(${CUDA_TEST_SOURCES} PROPERTIES LANGUAGE CUDA)

# # # #   # Add all tests
# # # #   add_ag_test(test_ag                tests/test_ag.cpp)
# # # #   add_ag_test(test_mlp               tests/test_mlp.cpp)
# # # #   add_ag_test(test_complex_mlp       tests/test_complex_mlp.cpp)
# # # #   add_ag_test(test_bench_relu        tests/bench_relu.cpp)
# # # #   add_ag_test(test_tiny_handcalc     tests/tiny_handcalc.cpp)
# # # #   add_ag_test(test_graph_compile     tests/test_graph_compile.cpp)
# # # #   add_ag_test(ag_core_test           tests/test.cpp)
# # # #   add_ag_test(test_checkpoint        tests/test_checkpoint.cpp)
# # # #   add_ag_test(test_inplace           tests/test_inplace.cpp)
# # # #   add_ag_test(test_careful_deletion  tests/test_careful_deletion.cpp)
# # # #   add_ag_test(test_nn                tests/test_nn.cpp)
# # # #   add_ag_test(test_version           tests/test_version.cpp)
# # # #   add_ag_test(test_nodeops           tests/test_nodeops.cpp)
# # # #   add_ag_test(test_kernels_cpu       tests/test_kernels_cpu.cpp)
# # # #   add_ag_test(test_vjp_kernels       tests/test_end_to_end_gpu.cpp)
# # # #   add_ag_test(test_nn_mlp            tests/test_mlp_training.cpp)
# # # #   add_ag_test(test_kernels_gpu       tests/test_kernels_gpu.cpp)
# # # #   add_ag_test(test_graph_cuda        tests/test_graph_gpu.cpp)
# # # # endif()


# # # # ********************************************************************************************************

# # # # # message(STATUS "cgadimpl build mode: ${CMAKE_BUILD_TYPE}")
# # # # cmake_minimum_required(VERSION 3.20)
# # # # project(cgadimpl LANGUAGES CXX CUDA)

# # # # # --- Compiler, CUDA, and OpenMP Setup ---
# # # # set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA SMs" FORCE)
# # # # find_package(CUDAToolkit REQUIRED)
# # # # find_package(OpenMP REQUIRED)

# # # # # Use the robust "bait and switch" for C++20
# # # # set(CMAKE_CXX_STANDARD 20)
# # # # set(CMAKE_CXX_STANDARD_REQUIRED ON)
# # # # set(CMAKE_CUDA_STANDARD 17)
# # # # set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# # # # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")

# # # # set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# # # # # --- FIX START: Modern and Robust Find Logic for OwnTensor Library ---
# # # # set(OWNTENSOR_DIR ${CMAKE_SOURCE_DIR}/../tensor)

# # # # # 1. Find the include directory by looking for a key header file.
# # # # find_path(OWNTENSOR_INCLUDE_DIR
# # # #     NAMES TensorLib.h
# # # #     HINTS ${OWNTENSOR_DIR}/include
# # # # )

# # # # # 2. Find the library file itself.
# # # # find_library(OWNTENSOR_LIBRARY
# # # #     NAMES tensor
# # # #     HINTS ${OWNTENSOR_DIR}/lib
# # # # )

# # # # # 3. Check if both were found and create a modern IMPORTED library target.
# # # # if(OWNTENSOR_INCLUDE_DIR AND OWNTENSOR_LIBRARY)
# # # #     add_library(OwnTensor::tensor SHARED IMPORTED)
# # # #     set_target_properties(OwnTensor::tensor PROPERTIES
# # # #         IMPORTED_LOCATION "${OWNTENSOR_LIBRARY}"
# # # #         INTERFACE_INCLUDE_DIRECTORIES "${OWNTENSOR_INCLUDE_DIR}"
# # # #     )
# # # #     message(STATUS "Found OwnTensor headers: ${OWNTENSOR_INCLUDE_DIR}")
# # # #     message(STATUS "Found OwnTensor library: ${OWNTENSOR_LIBRARY}")
# # # # else()
# # # #     # The original error message is still valid if they are not found.
# # # #     message(FATAL_ERROR "Could not find the OwnTensor library. Searched in ${OWNTENSOR_DIR}")
# # # # endif()
# # # # # --- FIX END ---

# # # # # ---- Sources ----
# # # # file(GLOB_RECURSE CGADIMPL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# # # # # ---- Library Definition ----
# # # # add_library(cgadimpl STATIC ${CGADIMPL_SRC})
# # # # add_library(cgadimpl::cgadimpl ALIAS cgadimpl)

# # # # # --- Define Include Directories ---
# # # # target_include_directories(cgadimpl PUBLIC
# # # #   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
# # # # )

# # # # # --- Define Library Links ---
# # # # # This propagates the dependency to any executable that links against cgadimpl.
# # # # target_link_libraries(cgadimpl PUBLIC OwnTensor::tensor dl)

# # # # # ---- Tests ----
# # # # if(true) # Simplified for clarity
# # # #   enable_testing()

# # # #   function(add_ag_test name src)
# # # #     add_executable(${name} ${src})
# # # #     # Link the test against our library, CUDA, and OpenMP
# # # #     target_link_libraries(${name} PRIVATE cgadimpl::cgadimpl CUDA::cudart OpenMP::OpenMP_CXX)
# # # #     add_test(NAME ${name} COMMAND ${name})
# # # #   endfunction()
  
# # # #   # ... (The rest of your test definitions are correct and do not need to change) ...
# # # #   set(CUDA_TEST_SOURCES
# # # #       tests/test_kernels_gpu.cpp
# # # #       tests/test_graph_gpu.cpp
# # # #       tests/test_end_to_end_gpu.cpp
# # # #   )
# # # #   set_source_files_properties(${CUDA_TEST_SOURCES} PROPERTIES LANGUAGE CUDA)

# # # #   add_ag_test(test_ag                tests/test_ag.cpp)
# # # #   add_ag_test(test_mlp               tests/test_mlp.cpp)
# # # #   add_ag_test(test_complex_mlp       tests/test_complex_mlp.cpp)
# # # #   add_ag_test(test_bench_relu        tests/bench_relu.cpp)
# # # #   add_ag_test(test_tiny_handcalc     tests/tiny_handcalc.cpp)
# # # #   add_ag_test(test_graph_compile     tests/test_graph_compile.cpp)
# # # #   add_ag_test(ag_core_test           tests/test.cpp)
# # # #   add_ag_test(test_checkpoint        tests/test_checkpoint.cpp)
# # # #   add_ag_test(test_inplace           tests/test_inplace.cpp)
# # # #   add_ag_test(test_careful_deletion  tests/test_careful_deletion.cpp)
# # # #   add_ag_test(test_nn                tests/test_nn.cpp)
# # # #   add_ag_test(test_version           tests/test_version.cpp)
# # # #   add_ag_test(test_nodeops           tests/test_nodeops.cpp)
# # # #   add_ag_test(test_kernels_cpu       tests/test_kernels_cpu.cpp)
# # # #   add_ag_test(test_vjp_kernels       tests/test_end_to_end_gpu.cpp)
# # # #   add_ag_test(test_nn_mlp            tests/test_mlp_training.cpp)
# # # #   add_ag_test(test_kernels_gpu       tests/test_kernels_gpu.cpp)
# # # #   add_ag_test(test_graph_cuda        tests/test_graph_gpu.cpp)
# # # # endif()

# # # # message(STATUS "cgadimpl build mode: ${CMAKE_BUILD_TYPE}")

# # # # ******************************************************************************************************************************
# # # # ====================================================================
# # # # FILE: cgadimpl/CMakeLists.txt (Definitive Corrected Version)
# # # # ====================================================================
# # # cmake_minimum_required(VERSION 3.20)
# # # project(cgadimpl LANGUAGES CXX CUDA)

# # # # --- Compiler, CUDA, and OpenMP Setup ---
# # # set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA SMs" FORCE)
# # # find_package(CUDAToolkit REQUIRED)
# # # find_package(OpenMP REQUIRED)

# # # # Use the robust "bait and switch" for C++20
# # # set(CMAKE_CXX_STANDARD 20)
# # # set(CMAKE_CXX_STANDARD_REQUIRED ON)
# # # set(CMAKE_CUDA_STANDARD 17)
# # # set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# # # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")

# # # set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# # # # --- Find and Link the Pre-built Tensor Library (Corrected Logic) ---
# # # set(OWNTENSOR_DIR ${CMAKE_SOURCE_DIR}/../tensor)

# # # # 1. Find the include directory by looking for a key header file.
# # # find_path(OWNTENSOR_INCLUDE_DIR
# # #     NAMES TensorLib.h
# # #     HINTS ${OWNTENSOR_DIR}/include
# # # )

# # # # 2. Find the library file itself.
# # # find_library(OWNTENSOR_LIBRARY
# # #     NAMES tensor
# # #     HINTS ${OWNTENSOR_DIR}/lib
# # # )

# # # # 3. Check if both were found and create a modern IMPORTED library target.
# # # if(OWNTENSOR_INCLUDE_DIR AND OWNTENSOR_LIBRARY)
# # #     add_library(OwnTensor::tensor SHARED IMPORTED)
# # #     set_target_properties(OwnTensor::tensor PROPERTIES
# # #         IMPORTED_LOCATION "${OWNTENSOR_LIBRARY}"
# # #         INTERFACE_INCLUDE_DIRECTORIES "${OWNTENSOR_INCLUDE_DIR}"
# # #     )
# # #     message(STATUS "Found OwnTensor headers: ${OWNTENSOR_INCLUDE_DIR}")
# # #     message(STATUS "Found OwnTensor library: ${OWNTENSOR_LIBRARY}")
# # # else()
# # #     message(FATAL_ERROR "Could not find the OwnTensor library. Searched in ${OWNTENSOR_DIR}")
# # # endif()


# # # # ---- Sources ----
# # # file(GLOB_RECURSE CGADIMPL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# # # # ---- Library Definition ----
# # # add_library(cgadimpl STATIC ${CGADIMPL_SRC})
# # # add_library(cgadimpl::cgadimpl ALIAS cgadimpl)

# # # # --- FIX: Add the WITH_CUDA compile definition ---
# # # # This makes CUDA-gated functions from the tensor library visible to cgadimpl.
# # # target_compile_definitions(cgadimpl PUBLIC WITH_CUDA)

# # # # --- Define Include Directories ---
# # # target_include_directories(cgadimpl PUBLIC
# # #   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
# # # )

# # # # --- Define Library Links ---
# # # # This links cgadimpl against the tensor library and propagates the dependency.
# # # target_link_libraries(cgadimpl PUBLIC OwnTensor::tensor dl)

# # # # ---- Tests (optional) ----
# # # if(true)
# # #   enable_testing()

# # #   function(add_ag_test name src)
# # #     add_executable(${name} ${src})
# # #     # --- FIX: Link tests against OpenMP in addition to other libraries ---
# # #     target_link_libraries(${name} PRIVATE cgadimpl::cgadimpl CUDA::cudart OpenMP::OpenMP_CXX)
# # #     add_test(NAME ${name} COMMAND ${name})
# # #   endfunction()
  
# # #   set(CUDA_TEST_SOURCES
# # #       tests/test_kernels_gpu.cpp
# # #       tests/test_graph_gpu.cpp
# # #       tests/test_end_to_end_gpu.cpp
# # #   )
# # #   set_source_files_properties(${CUDA_TEST_SOURCES} PROPERTIES LANGUAGE CUDA)

# # #   # Add all tests
# # #   add_ag_test(test_ag                tests/test_ag.cpp)
# # #   add_ag_test(test_mlp               tests/test_mlp.cpp)
# # #   add_ag_test(test_complex_mlp       tests/test_complex_mlp.cpp)
# # #   add_ag_test(test_bench_relu        tests/bench_relu.cpp)
# # #   add_ag_test(test_tiny_handcalc     tests/tiny_handcalc.cpp)
# # #   add_ag_test(test_graph_compile     tests/test_graph_compile.cpp)
# # #   add_ag_test(ag_core_test           tests/test.cpp)
# # #   add_ag_test(test_checkpoint        tests/test_checkpoint.cpp)
# # #   add_ag_test(test_inplace           tests/test_inplace.cpp)
# # #   add_ag_test(test_careful_deletion  tests/test_careful_deletion.cpp)
# # #   add_ag_test(test_nn                tests/test_nn.cpp)
# # #   add_ag_test(test_version           tests/test_version.cpp)
# # #   add_ag_test(test_nodeops           tests/test_nodeops.cpp)
# # #   add_ag_test(test_kernels_cpu       tests/test_kernels_cpu.cpp)
# # #   add_ag_test(test_vjp_kernels       tests/test_end_to_end_gpu.cpp)
# # #   add_ag_test(test_nn_mlp            tests/test_mlp_training.cpp)
# # #   add_ag_test(test_kernels_gpu       tests/test_kernels_gpu.cpp)
# # #   add_ag_test(test_graph_cuda        tests/test_graph_gpu.cpp)
# # # endif()

# # # message(STATUS "cgadimpl build mode: ${CMAKE_BUILD_TYPE}")
# # cmake_minimum_required(VERSION 3.20)
# # project(cgadimpl LANGUAGES CXX CUDA)

# # set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA SMs" FORCE)
# # find_package(CUDAToolkit REQUIRED)
# # find_package(OpenMP REQUIRED)

# # set(CMAKE_CXX_STANDARD 20)
# # set(CMAKE_CXX_STANDARD_REQUIRED ON)
# # set(CMAKE_CUDA_STANDARD 17)
# # set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")

# # set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# # set(OWNTENSOR_DIR ${CMAKE_SOURCE_DIR}/../tensor)
# # find_path(OWNTENSOR_INCLUDE_DIR NAMES TensorLib.h HINTS ${OWNTENSOR_DIR}/include)
# # find_library(OWNTENSOR_LIBRARY NAMES tensor HINTS ${OWNTENSOR_DIR}/lib)

# # if(OWNTENSOR_INCLUDE_DIR AND OWNTENSOR_LIBRARY)
# #     add_library(OwnTensor::tensor SHARED IMPORTED)
# #     set_target_properties(OwnTensor::tensor PROPERTIES
# #         IMPORTED_LOCATION "${OWNTENSOR_LIBRARY}"
# #         INTERFACE_INCLUDE_DIRECTORIES "${OWNTENSOR_INCLUDE_DIR}"
# #     )
# #     message(STATUS "Found OwnTensor headers: ${OWNTENSOR_INCLUDE_DIR}")
# #     message(STATUS "Found OwnTensor library: ${OWNTENSOR_LIBRARY}")
# # else()
# #     message(FATAL_ERROR "Could not find the OwnTensor library. Searched in ${OWNTENSOR_DIR}")
# # endif()

# # file(GLOB_RECURSE CGADIMPL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# # add_library(cgadimpl STATIC ${CGADIMPL_SRC})
# # add_library(cgadimpl::cgadimpl ALIAS cgadimpl)

# # target_compile_definitions(cgadimpl PUBLIC WITH_CUDA)
# # target_include_directories(cgadimpl PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
# # target_link_libraries(cgadimpl PUBLIC OwnTensor::tensor dl)

# # if(true)
# #   enable_testing()
# #   function(add_ag_test name src)
# #     add_executable(${name} ${src})
# #     target_link_libraries(${name} PRIVATE cgadimpl::cgadimpl CUDA::cudart OpenMP::OpenMP_CXX)
# #     add_test(NAME ${name} COMMAND ${name})
# #   endfunction()
  
# #   set(CUDA_TEST_SOURCES tests/test_kernels_gpu.cpp tests/test_graph_gpu.cpp tests/test_end_to_end_gpu.cpp)
# #   set_source_files_properties(${CUDA_TEST_SOURCES} PROPERTIES LANGUAGE CUDA)

# #   add_ag_test(test_ag tests/test_ag.cpp)
# #   add_ag_test(test_mlp tests/test_mlp.cpp)
# #   add_ag_test(test_complex_mlp tests/test_complex_mlp.cpp)
# #   add_ag_test(test_bench_relu tests/bench_relu.cpp)
# #   add_ag_test(test_tiny_handcalc tests/tiny_handcalc.cpp)
# #   add_ag_test(test_graph_compile tests/test_graph_compile.cpp)
# #   add_ag_test(ag_core_test tests/test.cpp)
# #   add_ag_test(test_checkpoint tests/test_checkpoint.cpp)
# #   add_ag_test(test_inplace tests/test_inplace.cpp)
# #   add_ag_test(test_careful_deletion tests/test_careful_deletion.cpp)
# #   add_ag_test(test_nn tests/test_nn.cpp)
# #   add_ag_test(test_version tests/test_version.cpp)
# #   add_ag_test(test_nodeops tests/test_nodeops.cpp)
# #   add_ag_test(test_kernels_cpu tests/test_kernels_cpu.cpp)
# #   add_ag_test(test_vjp_kernels tests/test_end_to_end_gpu.cpp)
# #   add_ag_test(test_nn_mlp tests/test_mlp_training.cpp)
# #   add_ag_test(test_kernels_gpu tests/test_kernels_gpu.cpp)
# #   add_ag_test(test_graph_cuda tests/test_graph_gpu.cpp)
# # endif()

# # message(STATUS "cgadimpl build mode: ${CMAKE_BUILD_TYPE}")
# cmake_minimum_required(VERSION 3.20)

# # --- FIX: Set the CUDA architecture BEFORE the project() command ---
# # This ensures CMake's initial compiler test can find the architecture.
# set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA SMs" FORCE)

# project(cgadimpl LANGUAGES CXX CUDA)

# # --- Compiler, CUDA, and OpenMP Setup ---
# find_package(CUDAToolkit REQUIRED)
# find_package(OpenMP REQUIRED)

# # Use the robust "bait and switch" for C++20
# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")

# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# # --- Find and Link the Pre-built Tensor Library (Corrected Logic) ---
# set(OWNTENSOR_DIR ${CMAKE_SOURCE_DIR}/../tensor)
# find_path(OWNTENSOR_INCLUDE_DIR NAMES TensorLib.h HINTS ${OWNTENSOR_DIR}/include)
# find_library(OWNTENSOR_LIBRARY NAMES tensor HINTS ${OWNTENSOR_DIR}/lib)

# if(OWNTENSOR_INCLUDE_DIR AND OWNTENSOR_LIBRARY)
#     add_library(OwnTensor::tensor SHARED IMPORTED)
#     set_target_properties(OwnTensor::tensor PROPERTIES
#         IMPORTED_LOCATION "${OWNTENSOR_LIBRARY}"
#         INTERFACE_INCLUDE_DIRECTORIES "${OWNTENSOR_INCLUDE_DIR}"
#     )
#     message(STATUS "Found OwnTensor headers: ${OWNTENSOR_INCLUDE_DIR}")
#     message(STATUS "Found OwnTensor library: ${OWNTENSOR_LIBRARY}")
# else()
#     message(FATAL_ERROR "Could not find the OwnTensor library. Searched in ${OWNTENSOR_DIR}")
# endif()

# # ---- Sources ----
# file(GLOB_RECURSE CGADIMPL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# # ---- Library Definition ----
# add_library(cgadimpl STATIC ${CGADIMPL_SRC})
# add_library(cgadimpl::cgadimpl ALIAS cgadimpl)

# target_compile_definitions(cgadimpl PUBLIC WITH_CUDA)
# target_include_directories(cgadimpl PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
# target_link_libraries(cgadimpl PUBLIC OwnTensor::tensor dl)

# # ---- Tests (optional) ----
# if(true)
#   enable_testing()
#   function(add_ag_test name src)
#     add_executable(${name} ${src})
#     target_link_libraries(${name} PRIVATE cgadimpl::cgadimpl CUDA::cudart OpenMP::OpenMP_CXX)
#     add_test(NAME ${name} COMMAND ${name})
#   endfunction()
  
#   set(CUDA_TEST_SOURCES tests/test_kernels_gpu.cpp tests/test_graph_gpu.cpp tests/test_end_to_end_gpu.cpp)
#   set_source_files_properties(${CUDA_TEST_SOURCES} PROPERTIES LANGUAGE CUDA)

#   add_ag_test(test_ag tests/test_ag.cpp)
#   add_ag_test(test_mlp tests/test_mlp.cpp)
#   add_ag_test(test_complex_mlp tests/test_complex_mlp.cpp)
#   add_ag_test(test_bench_relu tests/bench_relu.cpp)
#   add_ag_test(test_tiny_handcalc tests/tiny_handcalc.cpp)
#   add_ag_test(test_graph_compile tests/test_graph_compile.cpp)
#   add_ag_test(ag_core_test tests/test.cpp)
#   add_ag_test(test_checkpoint tests/test_checkpoint.cpp)
#   add_ag_test(test_inplace tests/test_inplace.cpp)
#   add_ag_test(test_careful_deletion tests/test_careful_deletion.cpp)
#   add_ag_test(test_nn tests/test_nn.cpp)
#   add_ag_test(test_version tests/test_version.cpp)
#   add_ag_test(test_nodeops tests/test_nodeops.cpp)
#   add_ag_test(test_kernels_cpu tests/test_kernels_cpu.cpp)
#   add_ag_test(test_vjp_kernels tests/test_end_to_end_gpu.cpp)
#   add_ag_test(test_nn_mlp tests/test_mlp_training.cpp)
#   add_ag_test(test_kernels_gpu tests/test_kernels_gpu.cpp)
#   add_ag_test(test_graph_cuda tests/test_graph_gpu.cpp)
# endif()

# message(STATUS "cgadimpl build mode: ${CMAKE_BUILD_TYPE}")
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
endif()

message(STATUS "cgadimpl build mode: ${CMAKE_BUILD_TYPE}")