

cmake_minimum_required(VERSION 3.20)

# --- FIX #2: Set the CUDA architecture BEFORE the project() command ---
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA SMs" FORCE)

project(cgadimpl LANGUAGES CXX CUDA)

set(MLIR_DIR "/home/blubridge-029/Desktop/llvm-project/build/lib/cmake/mlir")
set(mlir-compiler_DIR "/home/blubridge-029/Downloads/cgad/cgadimpl_/Nova-Compiler/install/lib/cmake/mlir-compiler")

# --- Compiler, CUDA, and OpenMP Setup ---
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# --- MLIR Setup ---
find_package(MLIR REQUIRED CONFIG)
find_package(mlir-compiler REQUIRED CONFIG)

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
# List source files explicitly for proper incremental compilation
# (similar to how the tensor team's Makefile uses pattern rules)
file(GLOB CGADIMPL_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB CGADIMPL_SRC_SUBDIRS "${CMAKE_CURRENT_SOURCE_DIR}/src/*/*.cpp")
set(CGADIMPL_SRC ${CGADIMPL_SRC_ROOT} ${CGADIMPL_SRC_SUBDIRS})

# Alternative: If you have deeper nesting, uncomment this and comment out the above:
# file(GLOB_RECURSE CGADIMPL_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# ---- Library Definition ----
add_library(cgadimpl STATIC ${CGADIMPL_SRC})
add_library(cgadimpl::cgadimpl ALIAS cgadimpl)

# target_compile_definitions(cgadimpl PUBLIC WITH_CUDA)
# target_include_directories(cgadimpl PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(cgadimpl PRIVATE /usr/local/cuda/include)
# target_link_libraries(cgadimpl PUBLIC OwnTensor::tensor dl)
target_compile_definitions(cgadimpl PUBLIC WITH_CUDA)
target_include_directories(cgadimpl PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(cgadimpl PUBLIC 
    OwnTensor::tensor 
    dl
    
    # Nova libraries (higher level first)
    MlirCompiler::NovaCompilerAPI
    MlirCompiler::NovaPipelines
    MlirCompiler::CompilerTransforms
    MlirCompiler::MLIRNova
    MlirCompiler::AffineTransforms
    
    # MLIR Conversion/Transforms
    MLIRReconcileUnrealizedCasts
    MLIRTosaToTensor
    MLIRTosaToLinalg
    MLIRTosaTransforms
    MLIRLinalgTransforms
    MLIRBufferizationTransforms
    MLIRBufferizationToMemRef
    MLIRBufferizationPipelines
    MLIRArithTransforms
    MLIRTensorTransforms
    MLIRSCFTransforms
    MLIRVectorTransforms
    MLIRMemRefTransforms
    
    # MLIR Dialects
    MLIRLinalgDialect
    MLIRBufferizationDialect
    MLIRTosaDialect
    MLIRArithDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRControlFlowDialect
    MLIRTensorDialect
    MLIRAffineDialect
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRUBDialect
    MLIRVectorDialect
    
    # Core Infrastructure
    MLIRIR
    MLIRPass
    MLIRSupport
    LLVMSupport
)
# --- FIX: Prioritize Local NovaCompiler Includes ---
# We derive the include path from the mlir-compiler_DIR which is passed as
# <install_prefix>/lib/cmake/mlir-compiler. We need <install_prefix>/include.
get_filename_component(NOVA_CMAKE_DIR "${mlir-compiler_DIR}" ABSOLUTE)
get_filename_component(NOVA_LIB_DIR "${NOVA_CMAKE_DIR}" DIRECTORY)
get_filename_component(NOVA_LIB_DIR "${NOVA_LIB_DIR}" DIRECTORY) # up to lib
get_filename_component(NOVA_INSTALL_PREFIX "${NOVA_LIB_DIR}" DIRECTORY) # up to install prefix
set(NOVA_INCLUDE_DIR "${NOVA_INSTALL_PREFIX}/include")

include_directories(BEFORE "${NOVA_INCLUDE_DIR}")
message(STATUS "Forcing NovaCompiler include dir: ${NOVA_INCLUDE_DIR}")
# ---- Tests ----
if(true)
  enable_testing()
  function(add_ag_test name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE cgadimpl::cgadimpl CUDA::cudart OpenMP::OpenMP_CXX)
    add_test(NAME ${name} COMMAND ${name})
  endfunction()
  
  set(CUDA_TEST_SOURCES Tests/test_graph_gpu.cpp Tests/test_end_to_end_gpu.cpp)
  set_source_files_properties(${CUDA_TEST_SOURCES} PROPERTIES LANGUAGE CUDA)

  add_ag_test(test_ag                    Tests/test_ag.cpp)
  add_ag_test(test_mlp                    Tests/test_mlp.cpp)
  add_ag_test(test_complex_mlp                    Tests/test_complex_mlp.cpp)
  # add_ag_test(test_bench_relu                    Tests/bench_relu.cpp)
  add_ag_test(test_tiny_handcalc                    Tests/tiny_handcalc.cpp)
  add_ag_test(test_graph_compile                    Tests/test_graph_compile.cpp)
  add_ag_test(ag_core_test                    Tests/test.cpp)
  add_ag_test(test_checkpoint                    Tests/test_checkpoint.cpp)
  add_ag_test(test_checkpoint_advanced           Tests/test_checkpoint_advanced.cpp)
  add_ag_test(test_inplace                    Tests/test_inplace.cpp)
  # add_ag_test(test_careful_deletion                    Tests/test_careful_deletion.cpp)
  add_ag_test(test_nn                    Tests/test_nn.cpp)
  # add_ag_test(test_version                    Tests/test_version.cpp)
  add_ag_test(test_nodeops                    Tests/test_nodeops.cpp)

  add_ag_test(test_vjp_kernels                    Tests/test_end_to_end_gpu.cpp)
  add_ag_test(test_nn_mlp                    Tests/test_mlp_training.cpp)

  add_ag_test(test_graph_cuda                    Tests/test_graph_gpu.cpp)
  add_ag_test(test_core_ops_gpu                    Tests/test_core_ops_gpu.cpp)
  add_ag_test(test_kitchen_sink_mlp                   Tests/test_kitchen_sink_mlp.cpp)
  # add_ag_test(test_gantlet                    Tests/test_gantlet.cpp)  #this works
  add_ag_test(test_scheme                    Tests/test_scheme.cpp)
  add_ag_test(test_phase1_metadata            Tests/test_phase1_metadata.cpp)  # Phase 1 critical metadata
  add_ag_test(test_phase1_advanced            Tests/test_phase1_advanced.cpp)  # Phase 1 advanced edge cases
  add_ag_test(test_dependency_counter                         Tests/test_dependency_counter.cpp )
  add_ag_test(test_dependncy_counter2                        Tests/test_dependncy_counter2.cpp)
  
  # Comprehensive dependency counter test suite
  add_ag_test(test_dependency_diamond            Tests/test_dependency_diamond.cpp)
  add_ag_test(test_dependency_wide_fanout        Tests/test_dependency_wide_fanout.cpp)
  add_ag_test(test_dependency_multi_join         Tests/test_dependency_multi_join.cpp)
  add_ag_test(test_dependency_correctness        Tests/test_dependency_correctness.cpp)
  add_ag_test(test_dependency_performance        Tests/test_dependency_performance.cpp)
  add_ag_test(test_dependency_weight_tying       Tests/test_dependency_weight_tying.cpp)
  add_ag_test(test_dependency_deep_branching     Tests/test_dependency_deep_branching.cpp)
  add_ag_test(time_estimation                   Tests/test_time_execution.cpp)
  add_ag_test(test_dag_analysis                Tests/test_dag_analysis.cpp)
  add_ag_test(test_warmup                      Tests/test_warmup.cpp)

  # Ops Tests
  add_ag_test(test_activation                  Tests/ops_test/test_activation.cpp)
  add_ag_test(test_arithmetic                  Tests/ops_test/test_arithmetic.cpp)
  add_ag_test(test_attention                   Tests/ops_test/test_attention.cpp)
  add_ag_test(test_linalg                      Tests/ops_test/test_linalg.cpp)
  add_ag_test(test_loss                        Tests/ops_test/test_loss.cpp)
  add_ag_test(test_normalization               Tests/ops_test/test_normalization.cpp)
  add_ag_test(test_reduction                   Tests/ops_test/test_reduction.cpp)
  add_ag_test(test_trigonometry                Tests/ops_test/test_trigonometry.cpp)

#Optim Tests
    add_ag_test(test_adam2                 Tests/test_adam2.cpp)
    add_ag_test(test_adam1                  Tests/test_adam1.cpp)
    add_ag_test(test_adam_optim_prod                  Tests/test_adam_optim_prod.cpp)

endif()

message(STATUS "cgadimpl build mode: ${CMAKE_BUILD_TYPE}")