cmake_minimum_required(VERSION 3.20)
project(cgadimpl LANGUAGES CXX)

# ---- Options ----
option(AG_PACKAGING    "Enable install + find_package exports" OFF)
option(AG_GLOB_SOURCES "Glob all .cpp under src/ (simplifies dev)" ON)
option(AG_BUILD_TESTS  "Build tests in tests/" ON)

# ---- Toolchain / common ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Sources ----
if(AG_GLOB_SOURCES)
  file(GLOB_RECURSE CGADIMPL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
else()
  # Explicit list fallback (keep as your teammate wrote)
  set(CGADIMPL_SRC
    src/core/graph.cpp
    src/core/ops.cpp
    src/core/schema.cpp
    src/core/debug.cpp
    src/core/export_hlo.cpp
    src/core/nodeops.cpp
    src/autodiff/autodiff.cpp
    src/autodiff/autodiff_vjp_ops.cpp
    src/autodiff/autodiff_jvp_ops.cpp
    src/tensor/tensor.cpp
    src/nn/nn.cpp
    src/optimizer/optim.cpp
    src/tools/kernels_loader.cpp
  )
endif()

if(NOT CGADIMPL_SRC)
  message(FATAL_ERROR "No source files found under ${CMAKE_CURRENT_SOURCE_DIR}/src")
endif()

# Sanity check (kernels ABI header)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/ad/kernels_api.hpp")
  message(FATAL_ERROR "Missing header: include/ad/kernels_api.hpp")
endif()

# ---- Library ----
add_library(cgadimpl ${CGADIMPL_SRC})
add_library(cgadimpl::cgadimpl ALIAS cgadimpl)

# Clean exported include dirs (fixes earlier export error)
target_include_directories(cgadimpl
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if (MSVC)
  target_compile_options(cgadimpl PRIVATE /W4 /permissive-)
else()
  target_compile_options(cgadimpl PRIVATE -Wall -Wextra -Wpedantic)
endif()

# dlopen on Linux
if(UNIX AND NOT APPLE)
  target_link_libraries(cgadimpl PRIVATE dl)
endif()

# ---- Tests (optional) ----
if(AG_BUILD_TESTS)
  include(CTest)
  enable_testing()

  function(add_ag_test name src)
  # Avoid CTest's reserved target name
  if("${name}" STREQUAL "test")
    set(name "ag_core_test")
    message(STATUS "Renaming test target 'test' -> '${name}' to avoid CTest clash")
  endif()
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE cgadimpl)
  if(UNIX AND NOT APPLE)
    target_link_libraries(${name} PRIVATE dl)
  endif()
  add_test(NAME ${name} COMMAND ${name})
  endfunction()


  # Your teammate's tests (checkpointing etc.)
  add_ag_test(test_ag            tests/test_ag.cpp)
  add_ag_test(test_mlp           tests/test_mlp.cpp)
  add_ag_test(test_complex_mlp   tests/test_complex_mlp.cpp)
  target_compile_definitions(test_complex_mlp PRIVATE AG_EXPOSE_AUTODIFF_RULES)
  add_ag_test(test_graph_compile tests/test_graph_compile.cpp)
  add_ag_test(ag_core_test      tests/test.cpp)
  add_ag_test(test_checkpoint    tests/test_checkpoint.cpp)
  # add_ag_test(smoke_all        tests/smoke_all.cpp)
endif()

# ---- Install & Packaging (only when enabled) ----
if(AG_PACKAGING)
  include(GNUInstallDirs)

  install(TARGETS cgadimpl
    EXPORT cgadimplTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfigVersion.cmake"
    VERSION 0.1.0
    COMPATIBILITY SameMajorVersion
  )

  # Template lives at: cgadimpl/cmake/cgadimplConfig.cmake.in
  #   @PACKAGE_INIT@
  #   include("${CMAKE_CURRENT_LIST_DIR}/cgadimplTargets.cmake")
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cgadimplConfig.cmake.in")
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cgadimplConfig.cmake.in"
      "${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfig.cmake" @ONLY
    )
    install(EXPORT cgadimplTargets
      NAMESPACE cgadimpl::
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cgadimpl
    )
    install(FILES
      "${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfig.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfigVersion.cmake"
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cgadimpl
    )
  else()
    message(WARNING "AG_PACKAGING=ON but cmake/cgadimplConfig.cmake.in not found; skipping package config.")
  endif()
endif()

# ---- Status ----
message(STATUS "cgadimpl build mode: ${CMAKE_BUILD_TYPE}")
message(STATUS "AG_PACKAGING: ${AG_PACKAGING}")
message(STATUS "AG_GLOB_SOURCES: ${AG_GLOB_SOURCES}")
message(STATUS "AG_BUILD_TESTS: ${AG_BUILD_TESTS}")
