add_executable(memcpy_vectorised
  kernels.cu
  main.cpp
)

target_include_directories(memcpy_vectorised
  PRIVATE
    ${CMAKE_SOURCE_DIR}/common/include
)

# Inherit warnings, C++20, CUDA flags, cudart, and common helpers
target_link_libraries(memcpy_vectorised
  PRIVATE
    # gpu_lab_common
    gpu_lab_options
)
