#include "gpu_lab/memory_resource.hpp"

#include <cassert>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>

#include <cuda_runtime.h>

#include "gpu_lab/cuda_check.hpp"
#include "gpu_lab/memory_location.hpp"

namespace {
  bool is_pow2(std::size_t x) noexcept { return x && (x & (x - 1)) == 0; }

  std::size_t round_up_pow2(std::size_t n, std::size_t align) noexcept {
    assert(n <= std::numeric_limits<std::size_t>::max() - (align - 1));
    return (n + (align - 1)) & ~(align - 1);
  }

  template <typename Resource>
  gpu_lab::detail::StridedBytes allocate_strided_bytes_with_resource(std::size_t block_bytes,
                                                                     std::size_t block_count,
                                                                     std::size_t block_alignment) {
    if (block_bytes == 0 || block_count == 0) {
      return {};
    }

    if (!is_pow2(block_alignment)) {
      throw std::invalid_argument{"allocate_strided_bytes: block_alignment must be power-of-two"};
    }

    const std::size_t stride_bytes = round_up_pow2(block_bytes, block_alignment);

    if (stride_bytes > (std::numeric_limits<size_t>::max() / block_count)) {
      throw std::bad_alloc{};
    }

    const std::size_t total_bytes = stride_bytes * block_count;

    void* ptr = Resource::allocate_bytes(total_bytes, block_alignment);
    return {ptr, stride_bytes};
  }
} // namespace

namespace gpu_lab::detail {
  void* HostMemoryResource::allocate_bytes(std::size_t nbytes, std::size_t alignment) {
    if (nbytes == 0) {
      return nullptr;
    } else if (alignment <= __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
      return ::operator new(nbytes);
    } else {
      return ::operator new(nbytes, std::align_val_t(alignment));
    }
  }

  StridedBytes HostMemoryResource::allocate_strided_bytes(std::size_t block_bytes,
                                                          std::size_t block_count,
                                                          std::size_t block_alignment) {
    return allocate_strided_bytes_with_resource<HostMemoryResource>(
        block_bytes, block_count, block_alignment);
  }

  void HostMemoryResource::deallocate_bytes(void* ptr,
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

  void* HostPinnedMemoryResource::allocate_bytes(std::size_t nbytes,
                                                 [[maybe_unused]] std::size_t alignment) {
    if (nbytes == 0) {
      return nullptr;
    }

    void* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, nbytes));
    return ptr;
  }

  StridedBytes HostPinnedMemoryResource::allocate_strided_bytes(std::size_t block_bytes,
                                                                std::size_t block_count,
                                                                std::size_t block_alignment) {
    return allocate_strided_bytes_with_resource<HostPinnedMemoryResource>(
        block_bytes, block_count, block_alignment);
  }

  void HostPinnedMemoryResource::deallocate_bytes(void* ptr,
                                                  [[maybe_unused]] std::size_t nbytes,
                                                  [[maybe_unused]] std::size_t alignment) noexcept {
    if (ptr) {
      CUDA_CHECK_TERMINATE(cudaFreeHost(ptr));
    }
  }

  void* DeviceMemoryResource::allocate_bytes(std::size_t nbytes,
                                             [[maybe_unused]] std::size_t alignment) {
    if (nbytes == 0) {
      return nullptr;
    }

    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, nbytes));
    return ptr;
  }

  StridedBytes DeviceMemoryResource::allocate_strided_bytes(std::size_t block_bytes,
                                                            std::size_t block_count,
                                                            [[maybe_unused]] std::size_t block_alignment) {
    if (block_bytes == 0 || block_count == 0) {
      return {};
    }

    StridedBytes res{};
    CUDA_CHECK(cudaMallocPitch(&res.ptr, &res.stride_bytes, block_bytes, block_count));
    return res;
  }

  void DeviceMemoryResource::deallocate_bytes(void* ptr,
                                              [[maybe_unused]] std::size_t nbytes,
                                              [[maybe_unused]] std::size_t alignment) noexcept {
    if (ptr) {
      CUDA_CHECK_TERMINATE(cudaFree(ptr));
    }
  }
} // namespace gpu_lab::detail
