cmake_minimum_required(VERSION 3.17)

project(DBF CXX C CUDA)

# ------------------------------------
# ----- Compiler settings ------------
# ------------------------------------
# C++17
set(CMAKE_CXX_STANDARD 17)

if(UNIX)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

IF(WIN32)
  add_definitions(-DNOMINMAX)
  add_compile_options(-MP)
ENDIF()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

option(DBF_BUILD_SANDBOX "Build the DBF Sandbox" ON)
option(DBF_CREATE_PYTHON_BINDINGS "Create Python Bindings" ON)

# ------------------------------------
# ----- Find dependent modules -------
# ------------------------------------
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

include(FetchContent)

# ----- Torch -----
FetchContent_Declare(Torch
  URL https://download.pytorch.org/libtorch/cu129/libtorch-shared-with-deps-2.8.0%2Bcu129.zip
)

FetchContent_MakeAvailable(Torch)
list(APPEND CMAKE_PREFIX_PATH "${torch_SOURCE_DIR}")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(DBF_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/dbf)

if(DBF_CREATE_PYTHON_BINDINGS)
  # ----- pybind11 -----
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.0.1
  )

  FetchContent_GetProperties(pybind11)

  if(NOT pybind11_POPULATED)
    FetchContent_MakeAvailable(pybind11)
  endif()

  find_package(Python REQUIRED COMPONENTS Interpreter Development REQUIRED)
endif()

add_subdirectory(dbf)

if(DBF_BUILD_SANDBOX)
  add_subdirectory(sandbox)
endif()

if(DBF_CREATE_PYTHON_BINDINGS)
  add_subdirectory(python)
endif()