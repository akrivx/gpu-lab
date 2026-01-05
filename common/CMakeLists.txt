add_library(gpu_lab_common
  src/experiment.cpp
  src/memory_resource.cpp
)

target_include_directories(gpu_lab_common
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(gpu_lab_common PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# Inherit all global options / CUDA runtime
target_link_libraries(gpu_lab_common
  PUBLIC
    gpu_lab_options
    gpu_lab_pitch_config
)
