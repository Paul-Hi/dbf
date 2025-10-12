# DBF - Differentiable Bilateral Filter

This repository offers a straightforward implementation of a Differentiable Bilateral Filter built with LibTorch, featuring Python bindings for seamless integration. The filter provides edge-preserving smoothing and is suitable for optimization or deep learning tasks that require differentiability. While custom CUDA kernels could further enhance performance and memory usage, this implementation prioritizes simplicity and broad compatibility with LibTorch-supported platforms.

## Features

- Simple bilateral filtering with support for gradients
- C++ Header-Only
- C++ core using LibTorch
- Python bindings using Pybind11

## Installation

### C++ (Header-Only or Build and Link)

You can use the filter as a header-only library by including the relevant headers in your project:

```cpp
#include "bilateral_filter.hpp"
```

Alternatively, you can build and link the library using CMake:

```cmake
add_subdirectory(dbf)
target_link_libraries(your_target PRIVATE dbf)
```

### Python

Install the Python bindings with pip:

```bash
pip install .
```

## Usage

Example usage can be found in the `sandbox` (C++) and `python` (Python) folders. These directories contain sample scripts demonstrating how to apply the differentiable bilateral filter in both languages.

## License

This project is licensed under the Apache License 2.0.
