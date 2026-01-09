#pragma once

#include <cstddef>

namespace gpu_lab::detail {
  // Layout description for a contiguous allocation divided into equally spaced
  // blocks.
  //
  // The allocation starts at `ptr`. Block `i` begins at
  // `ptr + i * stride_bytes`. `stride_bytes` is the byte distance between the
  // beginnings of consecutive blocks and may include padding to satisfy
  // alignment requirements.
  //
  // This type describes layout only; ownership and lifetime are managed
  // elsewhere.  
  struct StridedBytes {
    void* ptr = nullptr;
    std::size_t stride_bytes = 0;
  };
} // namespace gpu_lab::detail
