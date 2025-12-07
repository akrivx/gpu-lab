file(READ "${PITCH_FILE}" PITCH_VAL_RAW)
string(STRIP "${PITCH_VAL_RAW}" PITCH_VAL)

message(STATUS "GPU Lab: detected CUDA pitch alignment = ${PITCH_VAL}")

file(WRITE "${OUTPUT_FILE}" "
#pragma once
#include <cstddef>

/// Auto-generated at configure/build time.
/// Value taken from cudaDeviceProp::texturePitchAlignment (device 0).
/// GPU Lab assumes runtime execution on a compatible GPU.
namespace gpu_lab {
  inline constexpr std::size_t cuda_pitch_alignment = ${PITCH_VAL}u;
} // namespace gpu_lab
")
