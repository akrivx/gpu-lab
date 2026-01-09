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

    std::size_t total_bytes() const noexcept { return stride_bytes * block_count; }
  };

  template <ByteResource Resource>
  struct ByteStorageTraits {
    using allocation_type = ByteAllocation;

    static void deallocate(const allocation_type& alloc) noexcept {
      if (alloc.ptr) {
        Resource::deallocate_bytes(alloc.ptr, alloc.size_bytes, alloc.alignment);
      }
    }

    static allocation_type allocate(std::size_t nbytes, std::size_t alignment) {
      void* ptr = Resource::allocate_bytes(nbytes, alignment);
      return {ptr, nbytes, alignment};
    }
  };

  template <StridedByteResource Resource>
  struct StridedBytesStorageTraits {
    using allocation_type = StridedByteAllocation;

    static void deallocate(const allocation_type& alloc) noexcept {
      if (alloc.ptr) {
        Resource::deallocate_bytes(alloc.ptr, alloc.total_bytes(), alloc.alignment);
      }
    }

    static allocation_type
    allocate(std::size_t block_bytes, std::size_t block_count, std::size_t alignment) {
      StridedBytes layout = Resource::allocate_strided_bytes(block_bytes, block_count, alignment);
      return {layout.ptr, layout.stride_bytes, block_bytes, block_count, alignment};
    }
  };
} // namespace gpu_lab::detail
