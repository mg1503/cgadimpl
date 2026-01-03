


cmake_minimum_required(VERSION 3.2)

project(torch_llama_test)

set(CMAKE_PREFIX_PATH /cgadimpl/libtorch)

list(APPEND CMAKE_PREFIX_PATH "/home/ble-bridge022/Desktop/grishma/cgadimpl/libtorch")


find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} llama.cpp)
# add_executable( llama_test  llama.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC "${TORCH_LIBRARIES}")
# target_link_libraries(llama_test "${TORCH_LIBRARIES}")
target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})


set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 86)


# set_property(TARGET llama_test PROPERTY CXX_STANDARD 20)
