#pragma once

#include <cstddef>

namespace gpu_lab::detail {
  struct StridedBytes {
    void* ptr = nullptr;
    std::size_t stride_bytes = 0;
  };
} // namespace gpu_lab::detail
