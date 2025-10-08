cmake_minimum_required(VERSION 3.20)
project(agkernels_cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(${CGADIMPL_INCLUDE_DIR})

add_library(agkernels_cpu SHARED
  src/agkernels_cpu.cpp
)
set_target_properties(agkernels_cpu PROPERTIES OUTPUT_NAME "agkernels_cpu")

if (MSVC)
  target_compile_options(agkernels_cpu PRIVATE /W4 /permissive-)
else()
  target_compile_options(agkernels_cpu PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(GNUInstallDirs)
install(TARGETS agkernels_cpu
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
