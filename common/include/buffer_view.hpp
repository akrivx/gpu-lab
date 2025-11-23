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
    static constexpr MemoryLocation MEMORY_LOCATION = Loc;
    using element_type = T;
    using reference = T&;
    using const_reference = const T&;
    using value_type = std::remove_cv_t<T>;

    __host__ __device__ BufferView(T* data, size_t size) noexcept
      : data_{data}
      , size_{size}
    {}

    __host__ __device__ T* data() const noexcept { return data_; }

    __host__ __device__ size_t size() const noexcept { return size_; }
    __host__ __device__ bool empty() const noexcept { return size_ == 0; }

    __host__ __device__ const_reference operator[](size_t i) const noexcept
    {
      assert(i < size_);
      return data_[i];
    }
  
    __host__ __device__ reference operator[](size_t i) noexcept
    {
      assert(i < size_);
      return data_[i];
    }

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

    __host__ __device__ BufferView<const value_type, Loc> as_const() noexcept {
      return {data_, size_};
    }

    template<typename U>
    __host__ __device__ auto as() const noexcept { return rebind_view<U>(*this); }

    template<typename U>
    __host__ __device__ auto as() noexcept { return rebind_view<U>(*this); }
      
  private:
    template<typename U, typename View>
      requires ((!std::is_const_v<typename View::element_type> || std::is_const_v<U>) &&
                  sizeof(typename View::element_type) >= sizeof(U) &&
                  sizeof(typename View::element_type) % sizeof(U) == 0)
    static __host__ __device__ auto rebind_view(View v) {
      const auto nbytes = v.size() * sizeof(typename View::element_type);
      return BufferView<U, View::MEMORY_LOCATION>{reinterpret_cast<U*>(v.data()), nbytes / sizeof(U)};
    }

  private:
    T*     data_ = {};
    size_t size_ = {};
  };

  template<typename SrcT, MemoryLocation SrcLoc, typename DstT, MemoryLocation DstLoc>
    requires (std::is_same_v<std::remove_cv_t<SrcT>, DstT> &&
              !std::is_const_v<DstT>)
  void copy(
    BufferView<SrcT, SrcLoc> src,
    BufferView<DstT, DstLoc> dst)
  {
    if (src.size() != dst.size()) {
      throw std::runtime_error{"buffer copy size mismatch"};
    }

    CUDA_CHECK(
      cudaMemcpy(
        dst.data(),
        src.data(),
        src.size() * sizeof(SrcT),
        get_memcpy_kind<SrcLoc, DstLoc>()
      )
    );
  }

  template<typename SrcT, MemoryLocation SrcLoc, typename DstT, MemoryLocation DstLoc>
    requires ((SrcLoc == MemoryLocation::DEVICE || DstLoc == MemoryLocation::DEVICE) &&
              std::is_same_v<std::remove_cv_t<SrcT>, DstT> &&
              !std::is_const_v<DstT>)
  void copy_async(
    BufferView<SrcT, SrcLoc> src,
    BufferView<DstT, DstLoc> dst,
    cudaStream_t             stream = cudaStreamDefault)
  {
    if (src.size() != dst.size()) {
      throw std::runtime_error{"buffer copy async size mismatch"};
    }

    CUDA_CHECK(
      cudaMemcpyAsync(
        dst.data(),
        src.data(),
        src.size() * sizeof(SrcT),
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
} // namespace gpu_lab
