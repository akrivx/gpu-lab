#pragma once

#include <memory>
#include <type_traits>

#include <cuda_runtime.h>

#include "memory_location.hpp"
#include "cuda_check.hpp"

namespace gpu_lab {
  namespace detail {
    struct HostPinnedArrayDeleter {
      void operator()(void* host_ptr) const noexcept {
        cudaFreeHost(host_ptr);
      }
    };

    struct DeviceArrayDeleter {
      void operator()(void* dev_ptr) const noexcept {
        cudaFree(dev_ptr);
      }
    };
  } // namespace detail

  template<typename T, MemoryLocation Loc>
    requires (!std::is_array_v<T>)
  using UniqueArray = std::conditional_t<
    Loc == MemoryLocation::HOST_PAGEABLE,
    std::unique_ptr<T[]>,
    std::conditional_t<
      Loc == MemoryLocation::HOST_PINNED,
      std::unique_ptr<T[], detail::HostPinnedArrayDeleter>,
      std::unique_ptr<T[], detail::DeviceArrayDeleter>
    >
  >;

  template<typename T>
  using UniqueHostPageableArray = UniqueArray<T, MemoryLocation::HOST_PAGEABLE>;

  template<typename T>
  using UniqueHostPinnedArray = UniqueArray<T, MemoryLocation::HOST_PINNED>;

  template<typename T>
  using UniqueDeviceArray = UniqueArray<T, MemoryLocation::DEVICE>;

  template<typename T, MemoryLocation Loc>
  auto make_unique_array(size_t count) {
    if constexpr (Loc == MemoryLocation::HOST_PAGEABLE) {
      return UniqueHostPageableArray<T>{std::make_unique_for_overwrite<T[]>(count)};
    }
    else if constexpr (Loc == MemoryLocation::HOST_PINNED) {
      T* host_pinned_ptr = {};
      CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host_pinned_ptr), count * sizeof(T)));
      return UniqueHostPinnedArray<T>{host_pinned_ptr};
    }
    else { // MemoryLocation::DEVICE
      static_assert(Loc == MemoryLocation::DEVICE);
      T* dev_ptr = {};
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_ptr), count * sizeof(T)));
      return UniqueDeviceArray<T>{dev_ptr};
    }
  }

  template<typename T>
  auto make_unique_host_pageable_array(size_t count) {
    return make_unique_array<T, MemoryLocation::HOST_PAGEABLE>(count);
  }

  template<typename T>
  auto make_unique_host_pinned_array(size_t count) {
    return make_unique_array<T, MemoryLocation::HOST_PINNED>(count);
  }

  template<typename T>
  auto make_unique_device_array(size_t count) {
    return make_unique_array<T, MemoryLocation::DEVICE>(count);
  }

  template<typename T, MemoryLocation Loc>
  auto make_unique_array2d(size_t width, size_t height) {
    if constexpr (Loc == MemoryLocation::DEVICE) {
      T* dev_ptr   = {};
      size_t pitch = {};
      CUDA_CHECK(cudaMallocPitch(&dev_ptr, &pitch, width * sizeof(T), height));
      return std::make_pair(UniqueDeviceArray<T>{dev_ptr}, pitch);
    }
    else {
      size_t pitch = width * sizeof(T);
      return std::make_pair(make_unique_array<T, Loc>(width * height), pitch);
    }
  }

  template<typename T>
  auto make_unique_host_pageable_array2d(size_t width, size_t height) {
    return make_unique_array2d<T, MemoryLocation::HOST_PAGEABLE>(width, height);
  }

  template<typename T>
  auto make_unique_host_pinned_array2d(size_t width, size_t height) {
    return make_unique_array2d<T, MemoryLocation::HOST_PINNED>(width, height);
  }

  template<typename T>
  auto make_unique_device_array2d(size_t width, size_t height) {
    return make_unique_array2d<T, MemoryLocation::DEVICE>(width, height);
  }
} // namespace gpu_lab
