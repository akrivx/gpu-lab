#pragma once

#include "device_ptr.hpp"
#include "pinned_ptr.hpp"
#include "buffer_view.hpp"


namespace gpu_lab {

  namespace detail {

    template<typename T, MemoryLocation Loc>
    struct BufferTraits {
      using element_type = T;
      using handle_type = std::conditional_t<
        Loc == MemoryLocation::HOST_PAGEABLE,
        std::unique_ptr<T[]>,
        std::conditional_t<
          Loc == MemoryLocation::HOST_PINNED,
          UniquePinnedPtr<T[]>,
          UniqueDevicePtr<T[]>
        >
      >;

      static auto allocate(size_t size) {
        if constexpr (Loc == MemoryLocation::HOST_PAGEABLE) {
          return std::make_unique_for_overwrite<T[]>(size);
        }
        else if constexpr (Loc == MemoryLocation::HOST_PINNED) {
          return make_unique_pinned_ptr<T[]>(size);
        }
        else { // MemoryLocation::DEVICE
          static_assert(Loc == MemoryLocation::DEVICE);
          return make_unique_device_ptr<T[]>(size);
        }
      }
    };

  } // namespace detail

  template<typename T, MemoryLocation Loc>
  class Buffer {
    using traits = detail::BufferTraits<T, Loc>;
    
  public:
    using handle_type = typename traits::handle_type;
    using element_type = typename traits::element_type;

    Buffer() = default;

    Buffer(Buffer&&) noexcept = default;
    Buffer& operator=(Buffer&&) noexcept = default;

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    explicit Buffer(size_t size)
      : ptr_{traits::allocate(size)}
      , size_{size}
    {}

    Buffer(handle_type&& ptr, size_t size)
      : ptr_{std::move(ptr)}
      , size_{size}
    {}

    handle_type release() {
      auto ptr = std::exchange(ptr_, {});
      size_ = {};
      return std::move(ptr);
    }

    T* data() noexcept { return ptr_.get(); }
    const T* data() const noexcept { return ptr_.get(); }

    size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    auto view() const { return BufferView<const T, Loc>{data(), size()}; }
    auto view() { return BufferView<T, Loc>{data(), size()}; }
  
    operator BufferView<const T, Loc>() const { return view(); }
    operator BufferView<T, Loc>() { return view(); }

  private:
    handle_type ptr_ = {};
    size_t         size_ = {};
  };

  template<typename T, MemoryLocation Loc>
  auto view(const Buffer<T, Loc>& buf) noexcept {
    return BufferView<const T, Loc>{buf};
  }

  template<typename T, MemoryLocation Loc>
  auto view(Buffer<T, Loc>& buf) noexcept {
    return BufferView<T, Loc>{buf};
  }

  template<typename T>
  using DeviceBuffer = Buffer<T, MemoryLocation::DEVICE>;

  template<typename T>
  using HostPinnedBuffer = Buffer<T, MemoryLocation::HOST_PINNED>;

  template<typename T>
  using HostPageableBuffer = Buffer<T, MemoryLocation::HOST_PAGEABLE>;

  template<MemoryLocation Loc, typename T, MemoryLocation SrcLoc>
  auto clone(BufferView<T, SrcLoc> src) {
    Buffer<T, Loc> out{src.size()};
    copy(src, out);
    return out;
  }

  template<typename T, MemoryLocation Loc>
  auto to_device_buffer(BufferView<T, Loc> src) {
    return clone<MemoryLocation::DEVICE>(src);
  }

  template<typename T, MemoryLocation Loc>
  auto to_host_pageable_buffer(BufferView<T, Loc> src) {
    return clone<MemoryLocation::HOST_PAGEABLE>(src);
  }

  template<typename T, MemoryLocation Loc>
  auto to_host_pinned_buffer(BufferView<T, Loc> src) {
    return clone<MemoryLocation::HOST_PINNED>(src);
  }

  template<MemoryLocation Loc, typename T, MemoryLocation SrcLoc>
  auto clone_async(BufferView<T, SrcLoc> src, cudaStream_t stream = cudaStreamDefault) {
    Buffer<T, Loc> out{src.size()};
    copy_async(src, out);
    return out;
  }

  template<typename T, MemoryLocation Loc>
  auto to_device_buffer_async(BufferView<T, Loc> src, cudaStream_t stream = cudaStreamDefault) {
    return clone_async<MemoryLocation::DEVICE>(src, stream);
  }

  template<typename T, MemoryLocation Loc>
  auto to_host_pageable_buffer_async(BufferView<T, Loc> src, cudaStream_t stream = cudaStreamDefault) {
    return clone_async<MemoryLocation::HOST_PAGEABLE>(src, stream);
  }

  template<typename T, MemoryLocation Loc>
  auto to_host_pinned_buffer_async(BufferView<T, Loc> src, cudaStream_t stream = cudaStreamDefault) {
    return clone_async<MemoryLocation::HOST_PINNED>(src, stream);
  }

}
