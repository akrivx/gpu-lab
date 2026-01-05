#pragma once

#include <cstddef>
#include <new>
#include <type_traits>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "memory_location.hpp"

namespace gpu_lab::detail {
  struct HostMemoryResource {
    static constexpr MemoryLocation location = MemoryLocation::Host;

    static void* allocate_bytes(std::size_t nbytes, std::size_t alignment) {
      if (nbytes == 0) {
        return nullptr;
      } else if (alignment <= __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
        return ::operator new(nbytes);
      } else {
        return ::operator new(nbytes, std::align_val_t(alignment));
      }
    }

    static void deallocate_bytes(void* ptr,
                                 [[maybe_unused]] std::size_t nbytes,
                                 std::size_t alignment) noexcept {
      if (!ptr) {
        return;
      } else if (alignment <= __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
        ::operator delete(ptr);
      } else {
        ::operator delete(ptr, std::align_val_t(alignment));
      }
    }
  };

  struct HostPinnedMemoryResource {
    static constexpr MemoryLocation location = MemoryLocation::HostPinned;

    static void* allocate_bytes(std::size_t nbytes, [[maybe_unused]] std::size_t alignment) {
      if (nbytes == 0) {
        return nullptr;
      }

      void* ptr = nullptr;
      CUDA_CHECK(cudaMallocHost(&ptr, nbytes));
      return ptr;
    }

    static void deallocate_bytes(void* ptr,
                                 [[maybe_unused]] std::size_t nbytes,
                                 [[maybe_unused]] std::size_t alignment) noexcept {
      if (ptr) {
        CUDA_CHECK_TERMINATE(cudaFreeHost(ptr));
      }
    }
  };

  struct DeviceMemoryResource {
    static constexpr MemoryLocation location = MemoryLocation::Device;

    static void* allocate_bytes(std::size_t nbytes, [[maybe_unused]] std::size_t alignment) {
      if (nbytes == 0) {
        return nullptr;
      }

      void* ptr = nullptr;
      CUDA_CHECK(cudaMalloc(&ptr, nbytes));
      return ptr;
    }

    static void deallocate_bytes(void* ptr,
                                 [[maybe_unused]] std::size_t nbytes,
                                 [[maybe_unused]] std::size_t alignment) noexcept {
      if (ptr) {
        CUDA_CHECK_TERMINATE(cudaFree(ptr));
      }
    }
  };

  template <MemoryLocation Loc>
  struct DefaultMemoryResourceFor;

  template <>
  struct DefaultMemoryResourceFor<MemoryLocation::Host> {
    using type = HostMemoryResource;
  };

  template <>
  struct DefaultMemoryResourceFor<MemoryLocation::HostPinned> {
    using type = HostPinnedMemoryResource;
  };

  template <>
  struct DefaultMemoryResourceFor<MemoryLocation::Device> {
    using type = DeviceMemoryResource;
  };

  template <MemoryLocation Loc>
  using DefaultMemoryResource = DefaultMemoryResourceFor<Loc>::type;
} // namespace gpu_lab::detail
