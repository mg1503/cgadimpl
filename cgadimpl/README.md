# cgadimpl: Autodiff and Graph Implementation

`cgadimpl` is a C++20 library for automatic differentiation and computational graph operations, built on top of `TensorLib`. It supports dynamic graph construction, backpropagation, and execution on both CPU and GPU (via CUDA).

## Requirements

*   **CMake**: 3.20+
*   **C++ Compiler**: Supporting C++20 (e.g., GCC 10+, Clang 12+, MSVC 19.29+)
*   **CUDA Toolkit**: 11.0+ (Tested with 12.x/13.x)
*   **TensorLib**: Must be available as a sibling directory or installed.

## Directory Structure

*   **include/ad/**: Public headers for autodiff, graph types, and operations.
    *   `core/`: Core graph data structures (`Graph`, `Node`) and schema.
    *   `ops/`: Operation definitions (`NodeOps`, `Kernels`).
    *   `autodiff/`: Autodifferentiation triggers (`backward`), checkpointing, and memory optimization.
    *   `runtime/`: Runtime execution engines (e.g., CUDA Graphs).
    *   `utils/`: Debugging and export utilities.
*   **src/**: Source code matching the include structure.
    *   `core/`, `ops/`, `autodiff/`, `kernels/`, `runtime/`
*   **Tests/**: Unit tests and benchmarks.

## Building the Library

1.  **Create a build directory**:
    ```bash
    mkdir build
    cd build
    ```

2.  **Configure with CMake**:
    ```bash
    cmake ..
    ```

3.  **Build**:
    ```bash
    cmake --build . -j
    ```

## Running Tests

After building, you can run the tests using CTest or by executing the test binaries directly:

```bash
ctest --output-on-failure
```

or

```bash
./test_ag
./test_mlp
```

## Usage

Include the main header to access core functionality:

```cpp
#include "ad/ag_all.hpp"

// ... using ag::Graph, ag::Node, etc.
```

## Contributing

Please follow the directory structure when adding new files. New operations should have declarations in `include/ad/ops/` and implementations in `src/ops/`. Kernels go to `src/kernels/`.
