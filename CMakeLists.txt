cmake_minimum_required(VERSION 3.26)

project(gpu_lab
  VERSION 0.1.0
  LANGUAGES CXX CUDA
)

# -----------------------------
# Language standards
# -----------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# -----------------------------
# Global CUDA behaviour
# -----------------------------
if (NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# -----------------------------
# CUDA Toolkit (for CUDA::cudart, CUDA::cublas, etc.)
# -----------------------------
find_package(CUDAToolkit REQUIRED)

# -----------------------------
# Global "options target"
# -----------------------------
# This interface target holds all common compile options / features.
add_library(gpu_lab_options INTERFACE)

# C++20 everywhere
target_compile_features(gpu_lab_options INTERFACE cxx_std_20)

# Warnings & misc per compiler
if (MSVC)
  target_compile_options(gpu_lab_options INTERFACE
    /W4           # high warnings
    /permissive-  # stricter standard conformance
    /EHsc         # C++ exceptions
  )
else()
  target_compile_options(gpu_lab_options INTERFACE
    -Wall -Wextra -Wpedantic
  )
endif()

# CUDA-specific flags, only when compiling CUDA sources
target_compile_options(gpu_lab_options INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --use_fast_math>
)

# Everyone gets CUDA runtime by default
target_link_libraries(gpu_lab_options INTERFACE
  CUDA::cudart
)

# -----------------------------
# Subdirectories
# -----------------------------
# add_subdirectory(common)
add_subdirectory(experiments)
