#pragma once

#include <cstddef>
#include <type_traits>

#include "gpu_lab_config.hpp"  // generated in the build dir

namespace gpu_lab {
  template<class T>
  concept PitchedElement =
    std::is_trivially_copyable_v<T>
    && (cuda_pitch_alignment % sizeof(T) == 0);
} // namespace gpu_lab
