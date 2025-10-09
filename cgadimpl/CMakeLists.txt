# cmake_minimum_required(VERSION 3.16)
# project(ag LANGUAGES CXX)

# # ---- Options ---------------------------------------------------------------
# option(AG_BUILD_TOOLS "Build debug/export_hlo tools (src/tools)" ON)
# option(AG_ENABLE_WARNINGS "Enable -Wall/-Wextra/-Wpedantic" ON)
# option(AG_ENABLE_SANITIZERS "Enable ASan/UBSan (GNU/Clang)" OFF)

# # External tensor adapter toggles (only if/when you switch to the other team's Tensor)
# option(AG_USE_EXTERNAL_TENSOR "Use external Tensor implementation" OFF)
# set(AG_TENSOR_HEADER "" CACHE STRING "e.g. <Their/Path/Tensor.hpp>")
# set(AG_TENSOR_TYPE   "" CACHE STRING "e.g. their_ns::Tensor")

# # ---- Language / flags ------------------------------------------------------
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS ON) # matches -std=gnu++17 behavior

# if (AG_ENABLE_WARNINGS)
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

# if (AG_ENABLE_SANITIZERS AND (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU"))
#   add_compile_options(-fno-omit-frame-pointer -fsanitize=address,undefined)
#   add_link_options(    -fno-omit-frame-pointer -fsanitize=address,undefined)
# endif()

# # ---- Sources ---------------------------------------------------------------
# # Core + autodiff always included
# file(GLOB SRC_CORE      CONFIGURE_DEPENDS src/core/*.cpp)
# file(GLOB SRC_AUTODIFF  CONFIGURE_DEPENDS src/autodiff/*.cpp)
# file(GLOB SRC_TENSOR    CONFIGURE_DEPENDS src/tensor/*.cpp)
# file(GLOB SRC_NN        CONFIGURE_DEPENDS src/nn/*.cpp)
# file(GLOB SRC_OPTIMIZER CONFIGURE_DEPENDS src/optimizer/*.cpp) # <-- ADDED
# file(GLOB SRC_TOOLS     CONFIGURE_DEPENDS src/tools/*.cpp)

# #                                                                   ADDED SRC_OPTIMIZER
# #                                                                           |
# #                                                                           V
# set(SRC_ALL ${SRC_CORE} ${SRC_AUTODIFF} ${SRC_TENSOR} ${SRC_NN} ${SRC_OPTIMIZER} ${SRC_TOOLS})

# # Tools are optional (debug & StableHLO exporter)
# if (AG_BUILD_TOOLS)
#   file(GLOB SRC_TOOLS CONFIGURE_DEPENDS src/tools/*.cpp)
#   list(APPEND SRC_ALL ${SRC_TOOLS})
# endif()

# # ---- Library ---------------------------------------------------------------
# add_library(ag STATIC ${SRC_ALL})
# target_include_directories(ag PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# # External Tensor hook (matches your adapter in include/ag/tensor.hpp)
# if (AG_USE_EXTERNAL_TENSOR)
#   if (AG_TENSOR_HEADER STREQUAL "" OR AG_TENSOR_TYPE STREQUAL "")
#     message(FATAL_ERROR "Set AG_TENSOR_HEADER and AG_TENSOR_TYPE when AG_USE_EXTERNAL_TENSOR=ON")
#   endif()
#   target_compile_definitions(ag PUBLIC
#     AG_USE_EXTERNAL_TENSOR
#     AG_TENSOR_HEADER=${AG_TENSOR_HEADER}
#     AG_TENSOR_TYPE=${AG_TENSOR_TYPE}
#   )
# endif()

# # Nice output dirs
# set_target_properties(ag PROPERTIES
#   ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
#   LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
# )

# # ---- Tests -----------------------------------------------------------------
# function(add_ag_test name src)
#   add_executable(${name} ${src})
#   target_link_libraries(${name} PRIVATE ag)
#   target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
#   set_target_properties(${name} PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
#   )
# endfunction()

# # Add the tests you have
# add_ag_test(test_ag           tests/test_ag.cpp)
# add_ag_test(test_mlp          tests/test_mlp.cpp)
# add_ag_test(test_complex_mlp  tests/test_complex_mlp.cpp)
# target_compile_definitions(test_complex_mlp PRIVATE AG_EXPOSE_AUTODIFF_RULES)
# add_ag_test(test_graph_compile tests/test_graph_compile.cpp)
# add_ag_test(test              tests/test.cpp)
# # add_ag_test(smoke_all         tests/smoke_all.cpp)

#########################################################################################################################



# cmake_minimum_required(VERSION 3.20)
# project(cgadimpl LANGUAGES CXX)

# # ---- config ----
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# # ---- sources (glob everything under src/) ----
# file(GLOB_RECURSE CGADIMPL_SRC
#   "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
# )

# if(NOT CGADIMPL_SRC)
#   message(FATAL_ERROR "No source files found under ${CMAKE_CURRENT_SOURCE_DIR}/src")
# endif()

# # sanity: require the public headers root to exist
# if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/ad/kernels_api.hpp")
#   message(FATAL_ERROR "Missing header: include/ad/kernels_api.hpp")
# endif()

# # ---- library ----
# add_library(cgadimpl ${CGADIMPL_SRC})
# add_library(cgadimpl::cgadimpl ALIAS cgadimpl)
  
# target_include_directories(cgadimpl
#   PUBLIC
#      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#     $<INSTALL_INTERFACE:include>
# )

# if (MSVC)
#   target_compile_options(cgadimpl PRIVATE /W4 /permissive-)
# else()
#   target_compile_options(cgadimpl PRIVATE -Wall -Wextra -Wpedantic)
# endif()

# # dlopen on Linux
# if(UNIX AND NOT APPLE)
#   target_link_libraries(cgadimpl PRIVATE dl)
# endif()



# # ---- install (optional) ----
# include(GNUInstallDirs)

# install(TARGETS cgadimpl
#   EXPORT cgadimplTargets
#   ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#   RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
# )

# install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# # Package config (guarded so dev builds work even if template is absent)
# include(CMakePackageConfigHelpers)
# write_basic_package_version_file(
#   "${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfigVersion.cmake"
#   VERSION 0.1.0
#   COMPATIBILITY SameMajorVersion
# )

# if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cgadimplConfig.cmake.in")
#   configure_file(
#     "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cgadimplConfig.cmake.in"
#     "${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfig.cmake"
#     @ONLY
#   )
#   install(EXPORT cgadimplTargets
#     NAMESPACE cgadimpl::
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cgadimpl
#   )
#   install(FILES
#     "${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfig.cmake"
#     "${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfigVersion.cmake"
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cgadimpl
#   )
# endif()

##########################################################################################################################3
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

endfunction()

# Add the tests you have
add_ag_test(test_ag           tests/test_ag.cpp)
add_ag_test(test_mlp          tests/test_mlp.cpp)
add_ag_test(test_complex_mlp  tests/test_complex_mlp.cpp)
target_compile_definitions(test_complex_mlp PRIVATE AG_EXPOSE_AUTODIFF_RULES)
add_ag_test(test_graph_compile tests/test_graph_compile.cpp)
add_ag_test(test              tests/test.cpp)
add_ag_test(test_checkpoint    tests/test_checkpoint.cpp)
# add_ag_test(smoke_all         tests/smoke_all.cpp)

# ---- Install & Packaging ----
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfigVersion.cmake"
  VERSION 0.1.0
  COMPATIBILITY SameMajorVersion
)

# Template is tiny; create once: cgadimpl/cmake/cgadimplConfig.cmake.in
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
