# ====================================================================
# FILE: cgadimpl/CMakeLists.txt (Integrated with Pre-built Tensor Library)
# ====================================================================
cmake_minimum_required(VERSION 3.20)
project(cgadimpl LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA SMs" FORCE)

find_package(CUDAToolkit REQUIRED)

# --- Tensor Library Integration ---
set(OWNTENSOR_DIR ${CMAKE_SOURCE_DIR}/../tensor)
message(STATUS "Looking for OwnTensor library in: ${OWNTENSOR_DIR}")
include_directories(${OWNTENSOR_DIR}/include)
link_directories(${OWNTENSOR_DIR}/lib)

# --- Options ---
option(AG_PACKAGING    "Enable install + find_package exports" OFF)
option(AG_GLOB_SOURCES "Glob all .cpp under src/ (simplifies dev)" ON)
option(AG_BUILD_TESTS  "Build tests in tests/" ON)

# ===================== THIS IS THE DEFINITIVE FIX =====================
# We will "lie" to CMake's high-level feature detection and then force the
# correct flag at the low level.

# Command 1: Force all C++ files (.cpp) to compile with the C++20 standard.
# g++ is working correctly, so this is fine.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Command 2 (The Bait): Tell CMake's high-level system we want C++17 for CUDA.
# This will prevent the "does not know the compile flags" error.
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Command 3 (The Switch): Manually force the C++20 flag for the CUDA compiler.
# The compiler will see -std=c++17 followed by -std=c++20, and the last one wins.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")
# ====================================================================

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Sources ----
if(AG_GLOB_SOURCES)
  file(GLOB_RECURSE CGADIMPL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
endif()

# ---- Library ----
add_library(cgadimpl STATIC ${CGADIMPL_SRC})
add_library(cgadimpl::cgadimpl ALIAS cgadimpl)

target_include_directories(cgadimpl PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cgadimpl PUBLIC tensor dl)

# ---- Tests (optional) ----
if(AG_BUILD_TESTS)
  include(CTest)
  enable_testing()

  # This function now correctly links everything.
  # It only needs to link to cgadimpl. The dependencies on CUDA::cudart and
  # the 'tensor' library will be propagated automatically.
  function(add_ag_test name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE cgadimpl CUDA::cudart)
    add_test(NAME ${name} COMMAND ${name})
  endfunction()

  # Now all these calls will correctly link everything.
  add_ag_test(test_ag                tests/test_ag.cpp)
  add_ag_test(test_mlp               tests/test_mlp.cpp)
  add_ag_test(test_complex_mlp       tests/test_complex_mlp.cpp)
  target_compile_definitions(test_complex_mlp PRIVATE AG_EXPOSE_AUTODIFF_RULES)
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
  
  # Note: I have consolidated your test files for simplicity.
  # The 'add_ag_test' function is robust enough to handle them.
  add_ag_test(test_vjp_kernels       tests/test_end_to_end_gpu.cpp)
  add_ag_test(test_nn_mlp            tests/test_mlp_training.cpp)
  add_ag_test(test_kernels_gpu       tests/test_kernels_gpu.cpp)
  add_ag_test(test_graph_cuda        tests/test_graph_gpu.cpp)
  
  # We must still tell CMake to use NVCC for CUDA-heavy test files.
  set_source_files_properties(tests/test_kernels_gpu.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(tests/test_graph_gpu.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(tests/test_end_to_end_gpu.cpp PROPERTIES LANGUAGE CUDA)

endif()

# ... (rest of file is unchanged) ...
if(AG_PACKAGING)
  # ... (your packaging logic) ...
endif()
message(STATUS "cgadimpl build mode: ${CMAKE_BUILD_TYPE}")
message(STATUS "AG_PACKAGING: ${AG_PACKAGING}")
message(STATUS "AG_GLOB_SOURCES: ${AG_GLOB_SOURCES}")
message(STATUS "AG_BUILD_TESTS: ${AG_BUILD_TESTS}")