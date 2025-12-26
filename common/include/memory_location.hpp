#pragma once

#include <cuda_runtime.h>

namespace gpu_lab {
  enum class MemoryLocation { Host, HostPinned, Device };

  template <MemoryLocation SrcLoc, MemoryLocation DstLoc>
  constexpr cudaMemcpyKind get_memcpy_kind() {
    if constexpr (SrcLoc == MemoryLocation::Device) {
      if constexpr (DstLoc == MemoryLocation::Device) {
        return cudaMemcpyDeviceToDevice;
      } else {
        return cudaMemcpyDeviceToHost;
      }
    } else if constexpr (DstLoc == MemoryLocation::Device) {
      return cudaMemcpyHostToDevice;
    } else {
      return cudaMemcpyHostToHost;
    }
  }
} // namespace gpu_lab
