#pragma once

#include <cuda_runtime.h>

namespace gpu_lab {

  enum class MemoryLocation {
    HOST_PAGEABLE,
    HOST_PINNED,
    DEVICE
  };

  template<MemoryLocation SrcLoc, MemoryLocation DstLoc>
  constexpr cudaMemcpyKind get_memcpy_kind() {
    if constexpr (SrcLoc == MemoryLocation::DEVICE) {
      if constexpr (DstLoc == MemoryLocation::DEVICE) {
        return cudaMemcpyDeviceToDevice;
      }
      else {
        return cudaMemcpyDeviceToHost;
      }
    }
    else if constexpr (DstLoc == MemoryLocation::DEVICE) {
      return cudaMemcpyHostToDevice;
    }
    else {
      return cudaMemcpyHostToHost;
    }
  }

}