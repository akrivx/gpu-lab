#pragma once

#include <cstddef>
#include <cassert>
#include <stdexcept>
#include <type_traits>
#include "memory_location.hpp"


namespace gpu_lab {

  template<typename T, MemoryLocation Loc>
  class BufferView {
  public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;

    __host__ __device__ BufferView(T* data, size_t size) noexcept
      : data_{data}
      , size_{size}
    {}

    __host__ __device__ T* data() const noexcept { return data_; }

    __host__ __device__ size_t size() const noexcept { return size_; }
    __host__ __device__ bool empty() const noexcept { return size_ == 0; }

    __host__ __device__ BufferView subspan(size_t offset, size_t count) const noexcept {
      assert((offset + count) <= size_);
      return {data_ + offset, count};
    }

    __host__ __device__ BufferView take(size_t count) const noexcept {
      assert(count <= size_);
      return subspan(0, count);
    }

    __host__ __device__ BufferView drop(size_t count) const noexcept {
      assert(count <= size_);
      return subspan(count, size_ - count);
    }

  private:
    T*     data_ = {};
    size_t size_ = {};
  };

  template<typename T, MemoryLocation SrcLoc, MemoryLocation DstLoc>
  void copy(
    BufferView<const T, SrcLoc> src,
    BufferView<T, DstLoc>       dst
  )
  {
    if (src.size() != dst.size()) {
      throw std::runtime_error{"buffer copy size mismatch"};
    }

    CUDA_CHECK(
      cudaMemcpy(
        dst.data(),
        src.data(),
        src.size() * sizeof(T),
        get_memcpy_kind<SrcLoc, DstLoc>()
      )
    );
  }

  template<typename T, MemoryLocation SrcLoc, MemoryLocation DstLoc>
    requires (SrcLoc == MemoryLocation::DEVICE || DstLoc == MemoryLocation::DEVICE)
  void copy_async(
    BufferView<const T, SrcLoc> src,
    BufferView<T, DstLoc>       dst,
    cudaStream_t                stream = cudaStreamDefault
  )
  {
    if (src.size() != dst.size()) {
      throw std::runtime_error{"buffer copy async size mismatch"};
    }

    CUDA_CHECK(
      cudaMemcpyAsync(
        dst.data(),
        src.data(),
        src.size() * sizeof(T),
        get_memcpy_kind<SrcLoc, DstLoc>(),
        stream
      )
    );
  }

  template<typename T>
  using DeviceBufferView = BufferView<T, MemoryLocation::DEVICE>;

  template<typename T>
  using HostPinnedBufferView = BufferView<T, MemoryLocation::HOST_PINNED>;

  template<typename T>
  using HostPageableBufferView = BufferView<T, MemoryLocation::HOST_PAGEABLE>;

}
