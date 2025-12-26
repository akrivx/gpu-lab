#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "memory_location.hpp"

namespace gpu_lab {
  namespace detail {
    struct HostPinnedArrayDeleter {
      void operator()(void* host_ptr) const noexcept { cudaFreeHost(host_ptr); }
    };

    struct DeviceArrayDeleter {
      void operator()(void* dev_ptr) const noexcept { cudaFree(dev_ptr); }
    };
  } // namespace detail

  template <typename T, MemoryLocation Loc>
    requires(!std::is_array_v<T>)
  using UniqueArray =
      std::conditional_t<Loc == MemoryLocation::Host,
                         std::unique_ptr<T[]>,
                         std::conditional_t<Loc == MemoryLocation::HostPinned,
                                            std::unique_ptr<T[], detail::HostPinnedArrayDeleter>,
                                            std::unique_ptr<T[], detail::DeviceArrayDeleter>>>;

  template <typename T>
  using UniqueHostPageableArray = UniqueArray<T, MemoryLocation::Host>;

  template <typename T>
  using UniqueHostPinnedArray = UniqueArray<T, MemoryLocation::HostPinned>;

  template <typename T>
  using UniqueDeviceArray = UniqueArray<T, MemoryLocation::Device>;

  template <typename T, MemoryLocation Loc>
  auto make_unique_array(std::size_t count) {
    if constexpr (Loc == MemoryLocation::Host) {
      return UniqueHostPageableArray<T>{std::make_unique_for_overwrite<T[]>(count)};
    } else if constexpr (Loc == MemoryLocation::HostPinned) {
      T* HostPinned_ptr = {};
      CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&HostPinned_ptr), count * sizeof(T)));
      return UniqueHostPinnedArray<T>{HostPinned_ptr};
    } else // MemoryLocation::Device
    {
      static_assert(Loc == MemoryLocation::Device);
      T* dev_ptr = {};
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_ptr), count * sizeof(T)));
      return UniqueDeviceArray<T>{dev_ptr};
    }
  }

  template <typename T>
  auto make_unique_host_array(std::size_t count) {
    return make_unique_array<T, MemoryLocation::Host>(count);
  }

  template <typename T>
  auto make_unique_host_pinned_array(std::size_t count) {
    return make_unique_array<T, MemoryLocation::HostPinned>(count);
  }

  template <typename T>
  auto make_unique_device_array(std::size_t count) {
    return make_unique_array<T, MemoryLocation::Device>(count);
  }

  template <typename T, MemoryLocation Loc>
  auto make_unique_array2d(std::size_t height, std::size_t width) {
    if constexpr (Loc == MemoryLocation::Device) {
      T* dev_ptr = {};
      std::size_t pitch = {};
      CUDA_CHECK(cudaMallocPitch(&dev_ptr, &pitch, width * sizeof(T), height));
      return std::make_pair(UniqueDeviceArray<T>{dev_ptr}, pitch);
    } else {
      std::size_t pitch = width * sizeof(T);
      return std::make_pair(make_unique_array<T, Loc>(width * height), pitch);
    }
  }

  template <typename T>
  auto make_unique_host_array2d(std::size_t height, std::size_t width) {
    return make_unique_array2d<T, MemoryLocation::Host>(height, width);
  }

  template <typename T>
  auto make_unique_host_pinned_array2d(std::size_t height, std::size_t width) {
    return make_unique_array2d<T, MemoryLocation::HostPinned>(height, width);
  }

  template <typename T>
  auto make_unique_device_array2d(std::size_t height, std::size_t width) {
    return make_unique_array2d<T, MemoryLocation::Device>(height, width);
  }
} // namespace gpu_lab
