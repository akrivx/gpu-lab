function(gpu_lab_enable_pitch_config)
  message(STATUS "GPU Lab: pitch alignment auto-config enabled.")
  message(STATUS "GPU Lab: generated header will be: ${CMAKE_BINARY_DIR}/gpu_lab_config.hpp")

  # 1) Build the probe
  add_executable(gpu_lab_detect_pitch_alignment
    ${CMAKE_SOURCE_DIR}/cmake/detect_pitch_alignment.cu)

  set(PITCH_FILE "${CMAKE_BINARY_DIR}/cuda_pitch_alignment.txt")

  # 2) Run probe (device 0) to get texturePitchAlignment
  add_custom_command(
    OUTPUT "${PITCH_FILE}"
    COMMAND gpu_lab_detect_pitch_alignment > "${PITCH_FILE}"
    DEPENDS gpu_lab_detect_pitch_alignment
    COMMENT "Detecting CUDA texture pitch alignment for device 0"
  )

  # 3) Generate gpu_lab_config.hpp
  set(GPU_LAB_CONFIG_HDR "${CMAKE_BINARY_DIR}/gpu_lab_config.hpp")

  add_custom_command(
    OUTPUT "${GPU_LAB_CONFIG_HDR}"
    COMMAND ${CMAKE_COMMAND}
      -DOUTPUT_FILE="${GPU_LAB_CONFIG_HDR}"
      -DPITCH_FILE="${PITCH_FILE}"
      -P "${CMAKE_SOURCE_DIR}/cmake/generate_gpu_lab_config.cmake"
    DEPENDS "${PITCH_FILE}"
    COMMENT "Generating gpu_lab_config.hpp (auto pitch alignment)"
  )

  add_custom_target(gpu_lab_generate_config
    DEPENDS "${GPU_LAB_CONFIG_HDR}"
  )

  add_library(gpu_lab_pitch_config INTERFACE)
  add_dependencies(gpu_lab_pitch_config gpu_lab_generate_config)
  target_include_directories(gpu_lab_pitch_config
    INTERFACE ${CMAKE_BINARY_DIR}
  )
endfunction()
