cmake_minimum_required(VERSION 3.20)
project(agkernels LANGUAGES CXX)

set(CGADIMPL_INCLUDE_DIR "" CACHE PATH "Path to cgadimpl/include (contains ad/kernels_api.hpp)")
if(NOT CGADIMPL_INCLUDE_DIR)
  message(FATAL_ERROR "Set -DCGADIMPL_INCLUDE_DIR=/abs/path/to/cgadimpl/include")
endif()

add_subdirectory(cpu)
