cmake_minimum_required(VERSION 3.20)
project(mlir-compiler-test)

set(CMAKE_CXX_STANDARD 17)

# Find MLIR (required by mlir-compiler)
find_package(MLIR REQUIRED CONFIG)

# Find our mlir-compiler package
find_package(mlir-compiler REQUIRED CONFIG 
    PATHS /tmp/mlir-compiler-install/lib/cmake/mlir-compiler
)

# Create executable
add_executable(main main.cpp)

# Add include directories
target_include_directories(main PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    /tmp/mlir-compiler-install/include
)

# Link against required libraries
target_link_libraries(main PRIVATE
    MlirCompiler::MLIRNova
    MLIRIR
    MLIRSupport
    LLVMSupport
)
#cmake .. -DMLIR_DIR=/home/ble-bridge022/llvm-project/build/lib/cmake/mlir/