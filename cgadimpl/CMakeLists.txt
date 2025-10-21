# # =========================================================
# # FILE: cgadimpl/CMakeLists.txt (The NVCC Force Method)
# # =========================================================
# cmake_minimum_required(VERSION 3.20)
# project(cgadimpl LANGUAGES CXX CUDA)

# find_package(CUDAToolkit REQUIRED)

# # ---- Options ----
# option(AG_PACKAGING    "Enable install + find_package exports" OFF)
# option(AG_GLOB_SOURCES "Glob all .cpp under src/ (simplifies dev)" ON)
# option(AG_BUILD_TESTS  "Build tests in tests/" ON)

# # ---- Toolchain / common ----
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# # ---- Sources ----
# if(AG_GLOB_SOURCES)
#   file(GLOB_RECURSE CGADIMPL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
# endif()
# if(NOT CGADIMPL_SRC)
#   message(FATAL_ERROR "No source files found under ${CMAKE_CURRENT_SOURCE_DIR}/src")
# endif()

# # ---- Library ----
# add_library(cgadimpl ${CGADIMPL_SRC})
# add_library(cgadimpl::cgadimpl ALIAS cgadimpl)
# target_include_directories(cgadimpl PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>
# )
# if(UNIX AND NOT APPLE)
#   target_link_libraries(cgadimpl PRIVATE dl)
# endif()

# # ---- Tests (optional) ----
# if(AG_BUILD_TESTS)
#   include(CTest)
#   enable_testing()

#   function(add_ag_test name src)
#     if("${name}" STREQUAL "test")
#       set(name "ag_core_test")
#     endif()
#     add_executable(${name} ${src})
#     target_link_libraries(${name} PRIVATE cgadimpl)
#     if(UNIX AND NOT APPLE)
#       target_link_libraries(${name} PRIVATE dl)
#     endif()
#     add_test(NAME ${name} COMMAND ${name})
#   endfunction()

#   # All your CPU-only tests
#   add_ag_test(test_ag            tests/test_ag.cpp)
#   add_ag_test(test_mlp           tests/test_mlp.cpp)
#   add_ag_test(test_complex_mlp   tests/test_complex_mlp.cpp)
#   target_compile_definitions(test_complex_mlp PRIVATE AG_EXPOSE_AUTODIFF_RULES)
#   add_ag_test(test_bench_relu    tests/bench_relu.cpp)
#   add_ag_test(test_tiny_handcalc tests/tiny_handcalc.cpp)
#   add_ag_test(test_graph_compile tests/test_graph_compile.cpp)
#   add_ag_test(ag_core_test      tests/test.cpp)
#   add_ag_test(test_checkpoint    tests/test_checkpoint.cpp)
#   add_ag_test(test_inplace      tests/test_inplace.cpp)
#   add_ag_test(test_careful_deletion tests/test_careful_deletion.cpp)
#   add_ag_test(test_version           tests/test_version.cpp)
#   add_ag_test(test_nodeops          tests/test_nodeops.cpp)
#   add_ag_test(test_kernels_cpu tests/test_kernels_cpu.cpp)
  
#   # --- GPU Test Definition (THE FORCIBLE METHOD) ---
#   message(STATUS "Defining GPU kernel test target and forcing NVCC compiler")
#   add_executable(test_kernels_gpu tests/test_kernels_gpu.cpp)

#   # THIS IS THE MAGIC LINE:
#   # Tell CMake to treat this specific .cpp file as a CUDA file.
#   # This forces it to be compiled with NVCC, which will automatically handle linking.
#   set_source_files_properties(tests/test_kernels_gpu.cpp PROPERTIES LANGUAGE CUDA)

#   # Now we just need to link our core library. NVCC will handle the rest.
#   target_link_libraries(test_kernels_gpu PRIVATE cgadimpl)
  
#   add_test(NAME test_kernels_gpu COMMAND test_kernels_gpu)
#   # ---------------------------------------------

# endif()

# # ... (rest of file is unchanged, only removed some compile options for clarity) ...
# # ---- Install & Packaging ----
# if(AG_PACKAGING)
#   include(GNUInstallDirs)
#   install(TARGETS cgadimpl EXPORT cgadimplTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
#   install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
#   include(CMakePackageConfigHelpers)
#   write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfigVersion.cmake" VERSION 0.1.0 COMPATIBILITY SameMajorVersion)
#   if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cgadimplConfig.cmake.in")
#     configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/cgadimplConfig.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfig.cmake" @ONLY)
#     install(EXPORT cgadimplTargets NAMESPACE cgadimpl:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cgadimpl)
#     install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfigVersion.cmake" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cgadimpl)
#   else()
#     message(WARNING "AG_PACKAGING=ON but cmake/cgadimplConfig.cmake.in not found; skipping package config.")
#   endif()
# endif()
# message(STATUS "cgadimpl build mode: ${CMAKE_BUILD_TYPE}")
# message(STATUS "AG_PACKAGING: ${AG_PACKAGING}")
# message(STATUS "AG_GLOB_SOURCES: ${AG_GLOB_SOURCES}")
# message(STATUS "AG_BUILD_TESTS: ${AG_BUILD_TESTS}")




# ====================================================================
# FILE: cgadimpl/CMakeLists.txt (The Final, Unbreakable Fix)
# ====================================================================
cmake_minimum_required(VERSION 3.20)
project(cgadimpl LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# ---- Options ----
option(AG_PACKAGING    "Enable install + find_package exports" OFF)
option(AG_GLOB_SOURCES "Glob all .cpp under src/ (simplifies dev)" ON)
option(AG_BUILD_TESTS  "Build tests in tests/" ON)

# ---- Toolchain / common ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
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
  $<INSTALL_INTERFACE:include>
)

# ---- Tests (optional) ----
if(AG_BUILD_TESTS)
  include(CTest)
  enable_testing()

  # This function now contains the canonical and correct fix.
  function(add_ag_test name src)
    add_executable(${name} ${src})
    
    # ===================== THIS IS THE CORRECTED LINE =====================
    # We link against the canonical "CUDA::cudart" target, which is guaranteed
    # to exist when the CUDA language is enabled in the project() command.
    target_link_libraries(${name} PRIVATE cgadimpl CUDA::cudart dl)
    # ======================================================================

    add_test(NAME ${name} COMMAND ${name})
  endfunction()

  # Now all these calls will correctly link everything.
  add_ag_test(test_ag            tests/test_ag.cpp)
  add_ag_test(test_mlp           tests/test_mlp.cpp)
  add_ag_test(test_complex_mlp   tests/test_complex_mlp.cpp)
  target_compile_definitions(test_complex_mlp PRIVATE AG_EXPOSE_AUTODIFF_RULES)
  add_ag_test(test_bench_relu    tests/bench_relu.cpp)
  add_ag_test(test_tiny_handcalc tests/tiny_handcalc.cpp)
  add_ag_test(test_graph_compile tests/test_graph_compile.cpp)
  add_ag_test(ag_core_test      tests/test.cpp)
  add_ag_test(test_checkpoint    tests/test_checkpoint.cpp)
  add_ag_test(test_inplace      tests/test_inplace.cpp)
  add_ag_test(test_careful_deletion tests/test_careful_deletion.cpp)
  add_ag_test(test_version           tests/test_version.cpp)
  add_ag_test(test_nodeops          tests/test_nodeops.cpp)
  add_ag_test(test_kernels_cpu tests/test_kernels_cpu.cpp)
# --- Define the GPU test MANUALLY to force the CUDA compiler ---
add_executable(test_kernels_gpu tests/test_kernels_gpu.cpp)

# THIS IS THE ANSWER. This one line tells CMake to use NVCC for this file.
set_source_files_properties(tests/test_kernels_gpu.cpp PROPERTIES LANGUAGE CUDA)

# Now we link it manually.
target_link_libraries(test_kernels_gpu PRIVATE cgadimpl ${CUDAToolkit_LIBRARIES})
add_test(NAME test_kernels_gpu COMMAND test_kernels_gpu)

endif()

# ... (rest of file is unchanged) ...
if(AG_PACKAGING)
  include(GNUInstallDirs)
  install(TARGETS cgadimpl EXPORT cgadimplTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  include(CMakePackageConfigHelpers)
  write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfigVersion.cmake" VERSION 0.1.0 COMPATIBILITY SameMajorVersion)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cgadimplConfig.cmake.in")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/cgadimplConfig.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfig.cmake" @ONLY)
    install(EXPORT cgadimplTargets NAMESPACE cgadimpl:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cgadimpl)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfigVersion.cmake" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cgadimpl)
  else()
    message(WARNING "AG_PACKAGING=ON but cmake/cgadimplConfig.cmake.in not found; skipping package config.")
  endif()
endif()
message(STATUS "cgadimpl build mode: ${CMAKE_BUILD_TYPE}")
message(STATUS "AG_PACKAGING: ${AG_PACKAGING}")
message(STATUS "AG_GLOB_SOURCES: ${AG_GLOB_SOURCES}")
message(STATUS "AG_BUILD_TESTS: ${AG_BUILD_TESTS}")