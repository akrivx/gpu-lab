#pragma once

#include <cstddef>

#include "memory_resource_concepts.hpp"
#include "strided_bytes.hpp"

namespace gpu_lab::detail {
  struct ByteAllocation {
    void* ptr = nullptr;
    std::size_t size_bytes = 0;
    std::size_t alignment = 0;
  };

  struct StridedByteAllocation {
    void* ptr = nullptr;
    std::size_t stride_bytes = 0;
    std::size_t block_bytes = 0;
    std::size_t block_count = 0;
    std::size_t alignment = 0;
  };

  template <ByteResource Resource>
  struct ByteStorageTraits {
    using allocation_type = ByteAllocation;

    static inline constexpr bool is_strided = false;

    static void* data(const allocation_type& a) noexcept { return a.ptr; }
    static std::size_t size_bytes(const allocation_type& a) noexcept { return a.size_bytes; }

    static void deallocate(const allocation_type& a) noexcept {
      if (a.ptr) {
        Resource::deallocate_bytes(a.ptr, a.size_bytes, a.alignment);
      }
    }

    static allocation_type allocate(std::size_t nbytes, std::size_t alignment) {
      void* ptr = Resource::allocate_bytes(nbytes, alignment);
      return {ptr, nbytes, alignment};
    }
  };

  template <StridedByteResource Resource>
  struct StridedByteStorageTraits {
    using allocation_type = StridedByteAllocation;

    static inline constexpr bool is_strided = true;

    static void* data(const allocation_type& a) noexcept { return a.ptr; }
    static std::size_t stride_bytes(const allocation_type& a) noexcept { return a.stride_bytes; }
    static std::size_t block_bytes(const allocation_type& a) noexcept { return a.block_bytes; }
    static std::size_t block_count(const allocation_type& a) noexcept { return a.block_count; }

    static std::size_t size_bytes(const allocation_type& a) noexcept {
      return a.block_bytes * a.block_count;
    }

    static void deallocate(const allocation_type& a) noexcept {
      if (a.ptr) {
        Resource::deallocate_bytes(a.ptr, size_bytes(a), a.alignment);
      }
    }

    static allocation_type
    allocate(std::size_t block_bytes, std::size_t block_count, std::size_t alignment) {
      StridedBytes layout = Resource::allocate_strided_bytes(block_bytes, block_count, alignment);
      return {layout.ptr, layout.stride_bytes, block_bytes, block_count, alignment};
    }
  };
} // namespace gpu_lab::detail
