#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "memory_location.hpp"

namespace gpu_lab {
  template <typename T, MemoryLocation Loc>
  class BufferView {
  public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;

    static constexpr MemoryLocation location() { return Loc; }

    __host__ __device__ BufferView(T* ptr, std::size_t n) noexcept
        : data_{ptr}
        , size_{n} {}

    __host__ __device__ element_type* data() const noexcept { return data_; }

    __host__ __device__ std::size_t size() const noexcept { return size_; }

    __host__ __device__ std::size_t size_bytes() const noexcept {
      return size_ * sizeof(value_type);
    }

    __host__ __device__ bool empty() const noexcept { return size_ == 0; }

    __host__ __device__ element_type& operator[](std::size_t i) const noexcept {
      assert(i < size_);
      return data_[i];
    }

    __host__ __device__ BufferView subview(std::size_t offset, std::size_t count) const noexcept {
      assert((offset + count) <= size_);
      return {data_ + offset, count};
    }

    __host__ __device__ BufferView first(std::size_t count) const noexcept {
      assert(count <= size_);
      return subview(0, count);
    }

    __host__ __device__ BufferView last(std::size_t count) const noexcept {
      assert(count <= size_);
      return subview(size_ - count, count);
    }

  private:
    T* data_ = {};
    std::size_t size_ = {};
  };

  namespace detail {
    // Cannot cast away const
    template <typename To, typename From>
    concept ViewCastConstCompatible = !std::is_const_v<From> || std::is_const_v<To>;

    // sizeof(To) must divide sizeof(From)
    template <typename To, typename From>
    concept ViewCastSizeCompatible = (sizeof(From) % sizeof(To)) == 0;

    // Source alignment satisfies target
    template <typename To, typename From>
    concept ViewCastAlignCompatible = alignof(From) >= alignof(To);

    template <typename To, typename From>
    concept ViewCastCompatible =
        ViewCastConstCompatible<To, From> && ViewCastSizeCompatible<To, From>
        && ViewCastAlignCompatible<To, From>;
  } // namespace detail

  template <typename To, typename From, MemoryLocation Loc>
    requires detail::ViewCastCompatible<To, From>
  __host__ __device__ auto view_cast(BufferView<From, Loc> v) {
    const std::size_t bytes = v.size_bytes();
    return BufferView<To, Loc>{reinterpret_cast<To*>(v.data()), bytes / sizeof(To)};
  }

  template <typename T, MemoryLocation Loc>
  __host__ __device__ BufferView<const std::byte, Loc> as_bytes(BufferView<T, Loc> v) {
    return view_cast<const std::byte>(v);
  }

  template <typename T, MemoryLocation Loc>
  __host__ __device__ BufferView<std::byte, Loc> as_writable_bytes(BufferView<T, Loc> v) {
    return view_cast<std::byte>(v);
  }

  template <typename T, MemoryLocation Loc>
  __host__ __device__ auto as_const(BufferView<T, Loc> v) {
    return BufferView<const std::remove_cv_t<T>, Loc>{v.data(), v.size()};
  }

  namespace detail {
    template <typename SrcView, typename DstView>
    concept ViewCopyCompatible =
        std::same_as<typename SrcView::value_type, typename DstView::value_type>
        && !std::is_const_v<typename DstView::element_type>;

    template <typename SrcView, typename DstView>
    concept ViewCopyAsyncCompatible = ViewCopyCompatible<SrcView, DstView>
                                      && (SrcView::location() == MemoryLocation::Device
                                          || DstView::location() == MemoryLocation::Device);

    template <typename SrcView, typename DstView>
      requires ViewCopyCompatible<SrcView, DstView>
    void copy_view(SrcView src, DstView dst) {
      if (src.size() != dst.size()) {
        throw std::runtime_error{"copy: size mismatch"};
      }
      CUDA_CHECK(cudaMemcpy(dst.data(),
                            src.data(),
                            src.size() * sizeof(typename SrcView::value_type),
                            get_memcpy_kind<SrcView::location(), DstView::location()>()));
    }

    template <typename SrcView, typename DstView>
      requires ViewCopyCompatible<SrcView, DstView>
    void copy_view_async(SrcView src, DstView dst, cudaStream_t stream) {
      if (src.size() != dst.size()) {
        throw std::runtime_error{"copy_async: size mismatch"};
      }
      CUDA_CHECK(cudaMemcpyAsync(dst.data(),
                                 src.data(),
                                 src.size() * sizeof(typename SrcView::value_type),
                                 get_memcpy_kind<SrcView::location(), DstView::location()>(),
                                 stream));
    }
  } // namespace detail

  template <typename SrcT, MemoryLocation SrcLoc, typename DstT, MemoryLocation DstLoc>
  void copy(BufferView<SrcT, SrcLoc> src, BufferView<DstT, DstLoc> dst) {
    detail::copy_view(src, dst);
  }

  template <typename SrcT, MemoryLocation SrcLoc, typename DstT, MemoryLocation DstLoc>
  void copy_async(BufferView<SrcT, SrcLoc> src,
                  BufferView<DstT, DstLoc> dst,
                  cudaStream_t stream = cudaStreamDefault) {
    detail::copy_view_async(src, dst, stream);
  }

  template <typename T>
  using HostBufferView = BufferView<T, MemoryLocation::Host>;

  template <typename T>
  using HostPinnedBufferView = BufferView<T, MemoryLocation::HostPinned>;

  template <typename T>
  using DeviceBufferView = BufferView<T, MemoryLocation::Device>;
} // namespace gpu_lab
