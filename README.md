# DBF - Differentiable Bilateral Filter

This repository offers a two implementations of a Differentiable Bilateral Filter built with LibTorch, featuring Python bindings for seamless integration. The filter provides edge-preserving smoothing and is suitable for optimization or deep learning tasks that require differentiability.
The core functionality is provided by two functions: `bilateralFilter` (`bilateral_filter` in Python) and `bilateralFilterCuda` (`bilateral_filter_cuda` in Python). The first uses basic Torch tensor operations for clarity, while the second leverages custom CUDA kernels for higher performance and lower VRAM usage.

## Features

- Simple bilateral filtering with support for gradients
- C++ core using LibTorch
- Python bindings using Pybind11

## Installation

Depending on your CUDA version, you may need to adjust how you fetch LibTorch in your top-level `CMakeLists.txt` for C++ and specify the correct PyTorch version in your `pyproject.toml` for Python.

**C++ (LibTorch FetchContent):**
In your `CMakeLists.txt`, set the appropriate LibTorch URL for your CUDA version. For example:

```cmake
FetchContent_Declare(
    torch
    URL https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-2.2.0.zip # Change cu118 to match your CUDA version
)
```
Refer to [PyTorch LibTorch download page](https://pytorch.org/get-started/locally/) for the correct URL.

**Python (`pyproject.toml`):**
Specify the PyTorch version and CUDA variant in your dependencies. For example:

```toml
[build-system]
requires = [
    "torch==2.2.0+cu118" # Change cu118 to match your CUDA version
]

dependencies = [
    "torch==2.2.0+cu118" # Change cu118 to match your CUDA version
]

[dependency-groups]
torch = ["torch==2.2.0+cu118"] # Change cu118 to match your CUDA version
```

See [PyTorch installation instructions](https://pytorch.org/get-started/locally/) for the correct version string.


### C++

You can build and link the library using CMake:

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
