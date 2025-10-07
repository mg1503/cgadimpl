cmake_minimum_required(VERSION 3.16)
project(ag LANGUAGES CXX)

# ---- Options ---------------------------------------------------------------
option(AG_BUILD_TOOLS "Build debug/export_hlo tools (src/tools)" ON)
option(AG_ENABLE_WARNINGS "Enable -Wall/-Wextra/-Wpedantic" ON)
option(AG_ENABLE_SANITIZERS "Enable ASan/UBSan (GNU/Clang)" OFF)

# External tensor adapter toggles (only if/when you switch to the other team's Tensor)
option(AG_USE_EXTERNAL_TENSOR "Use external Tensor implementation" OFF)
set(AG_TENSOR_HEADER "" CACHE STRING "e.g. <Their/Path/Tensor.hpp>")
set(AG_TENSOR_TYPE   "" CACHE STRING "e.g. their_ns::Tensor")

# ---- Language / flags ------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON) # matches -std=gnu++17 behavior

if (AG_ENABLE_WARNINGS)
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if (AG_ENABLE_SANITIZERS AND (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU"))
  add_compile_options(-fno-omit-frame-pointer -fsanitize=address,undefined)
  add_link_options(    -fno-omit-frame-pointer -fsanitize=address,undefined)
endif()

# ---- Sources ---------------------------------------------------------------
# Core + autodiff always included
file(GLOB SRC_CORE      CONFIGURE_DEPENDS src/core/*.cpp)
file(GLOB SRC_AUTODIFF  CONFIGURE_DEPENDS src/autodiff/*.cpp)
file(GLOB SRC_TENSOR    CONFIGURE_DEPENDS src/tensor/*.cpp)
file(GLOB SRC_NN        CONFIGURE_DEPENDS src/nn/*.cpp)
file(GLOB SRC_OPTIMIZER CONFIGURE_DEPENDS src/optimizer/*.cpp) # <-- ADDED
file(GLOB SRC_TOOLS     CONFIGURE_DEPENDS src/tools/*.cpp)

#                                                                   ADDED SRC_OPTIMIZER
#                                                                           |
#                                                                           V
set(SRC_ALL ${SRC_CORE} ${SRC_AUTODIFF} ${SRC_TENSOR} ${SRC_NN} ${SRC_OPTIMIZER} ${SRC_TOOLS})

# Tools are optional (debug & StableHLO exporter)
if (AG_BUILD_TOOLS)
  file(GLOB SRC_TOOLS CONFIGURE_DEPENDS src/tools/*.cpp)
  list(APPEND SRC_ALL ${SRC_TOOLS})
endif()

# ---- Library ---------------------------------------------------------------
add_library(ag STATIC ${SRC_ALL})
target_include_directories(ag PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# External Tensor hook (matches your adapter in include/ag/tensor.hpp)
if (AG_USE_EXTERNAL_TENSOR)
  if (AG_TENSOR_HEADER STREQUAL "" OR AG_TENSOR_TYPE STREQUAL "")
    message(FATAL_ERROR "Set AG_TENSOR_HEADER and AG_TENSOR_TYPE when AG_USE_EXTERNAL_TENSOR=ON")
  endif()
  target_compile_definitions(ag PUBLIC
    AG_USE_EXTERNAL_TENSOR
    AG_TENSOR_HEADER=${AG_TENSOR_HEADER}
    AG_TENSOR_TYPE=${AG_TENSOR_TYPE}
  )
endif()

# Nice output dirs
set_target_properties(ag PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# ---- Tests -----------------------------------------------------------------
function(add_ag_test name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE ag)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
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