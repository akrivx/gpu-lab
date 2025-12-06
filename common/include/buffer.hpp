#pragma once

#include <memory>
#include <utility>

#include <cuda_runtime.h>

#include "memory_location.hpp"
#include "unique_array.hpp"
#include "buffer_view.hpp"

namespace gpu_lab {
  template<typename T, MemoryLocation Loc>
  class [[nodiscard]] Buffer {
  public:
    using handle_type = UniqueArray<T, Loc>;
    using element_type = T;
    using view_type = BufferView<T, Loc>;
    using const_view_type = BufferView<const T, Loc>;

    static constexpr MemoryLocation location() { return Loc; }

    Buffer() noexcept = default;

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    explicit Buffer(size_t count)
      : handle_{make_unique_array<T, Loc>(count)}
      , size_{count}
    {}
  
    Buffer(Buffer&& o) noexcept
      : handle_{std::exchange(o.handle_, {})}
      , size_{std::exchange(o.size_, {})}
    {}
      
    Buffer& operator=(Buffer&& o) noexcept {
      if (this != &o) {
        handle_ = std::exchange(o.handle_, {});
        size_ = std::exchange(o.size_, {});
      }
      return *this;
    }

    handle_type release() noexcept {
      size_ = {};
      return std::exchange(handle_, {});
    }

    element_type* data() noexcept { return handle_.get(); }
    const element_type* data() const noexcept { return handle_.get(); }
    const element_type* cdata() const noexcept { return handle_.get(); }

    size_t size() const noexcept { return size_; }
    size_t size_bytes() const noexcept { return size_ * sizeof(element_type); }
    bool empty() const noexcept { return size_ == 0; }

    view_type view() noexcept { return {data(), size()}; }
    const_view_type view() const noexcept { return {data(), size()}; }
    const_view_type cview() const noexcept { return as_const(view()); }

  private:
    handle_type handle_ = {};
    size_t      size_   = {};
  };

  template<typename To, typename From, MemoryLocation Loc>
  auto view_as(const Buffer<From, Loc>& buf) noexcept {
    return view_cast<To>(buf.view());
  }

  template<typename To, typename From, MemoryLocation Loc>
  auto view_as(Buffer<From, Loc>& buf) noexcept {
    return view_cast<To>(buf.view());
  }

  template<typename T, MemoryLocation Loc>
  auto view_as_bytes(const Buffer<T, Loc>& buf) {
    return as_bytes(buf.view());
  }

  template<typename T, MemoryLocation Loc>
  auto view_as_writable_bytes(Buffer<T, Loc>& buf) {
    return as_writable_bytes(buf.view());
  }

  template<typename T>
  using HostBuffer = Buffer<T, MemoryLocation::Host>;

  template<typename T>
  using HostPinnedBuffer = Buffer<T, MemoryLocation::HostPinned>;

  template<typename T>
  using DeviceBuffer = Buffer<T, MemoryLocation::Device>;

  // Buffer copies

  template<typename T, MemoryLocation SrcLoc, MemoryLocation DstLoc>
  void copy(const Buffer<T, SrcLoc>& src, Buffer<T, DstLoc>& dst) {
    copy(src.view(), dst.view());
  }

  template<typename T, MemoryLocation SrcLoc, MemoryLocation DstLoc>
  void copy_async(
    const Buffer<T, SrcLoc>& src,
    Buffer<T, DstLoc>&       dst,
    cudaStream_t             stream = cudaStreamDefault)
  {
    copy_async(src.view(), dst.view(), stream);
  }

  // Copying a view into a newly created buffer

  template<MemoryLocation DstLoc, typename T, MemoryLocation SrcLoc>
  auto to_buffer(BufferView<T, SrcLoc> src) {
    using U = typename BufferView<T, SrcLoc>::value_type;
    Buffer<U, DstLoc> out{src.size()};
    copy(src, out.view());
    return out;
  }

  template<typename T, MemoryLocation Loc>
  auto to_host_buffer(BufferView<T, Loc> src) {
    return to_buffer<MemoryLocation::Host>(src);
  }

  template<typename T, MemoryLocation Loc>
  auto to_host_pinned_buffer(BufferView<T, Loc> src) {
    return to_buffer<MemoryLocation::HostPinned>(src);
  }

  template<typename T, MemoryLocation Loc>
  auto to_device_buffer(BufferView<T, Loc> src) {
    return to_buffer<MemoryLocation::Device>(src);
  }

  template<MemoryLocation DstLoc, typename T, MemoryLocation SrcLoc>
  auto to_buffer_async(BufferView<T, SrcLoc> src, cudaStream_t stream = cudaStreamDefault) {
    using U = typename BufferView<T, SrcLoc>::value_type;
    Buffer<U, DstLoc> out{src.size()};
    copy_async(src, out.view(), stream);
    return out;
  }

  template<typename T>
  auto to_host_pinned_buffer_async(DeviceBufferView<T> src, cudaStream_t stream = cudaStreamDefault) {
    return to_buffer_async<MemoryLocation::HostPinned>(src, stream);
  }

  template<typename T>
  auto to_device_buffer_async(HostPinnedBufferView<T> src, cudaStream_t stream = cudaStreamDefault) {
    return to_buffer_async<MemoryLocation::Device>(src, stream);
  }
  
  template<typename T>
  auto to_device_buffer_async(DeviceBufferView<T> src, cudaStream_t stream = cudaStreamDefault) {
    return to_buffer_async<MemoryLocation::Device>(src, stream);
  }

  // Copying a buffer into a newly created buffer

  template<MemoryLocation DstLoc, typename T, MemoryLocation SrcLoc>
  auto to_buffer(const Buffer<T, SrcLoc>& src) {
    return to_buffer<DstLoc>(src.view());
  }

  template<typename T, MemoryLocation Loc>
  auto to_host_buffer(const Buffer<T, Loc>& src) {
    return to_host_buffer(src.view());
  }

  template<typename T, MemoryLocation Loc>
  auto to_host_pinned_buffer(const Buffer<T, Loc>& src) {
    return to_host_pinned_buffer(src.view());
  }

  template<typename T, MemoryLocation Loc>
  auto to_device_buffer(const Buffer<T, Loc>& src) {
    return to_device_buffer(src.view());
  }

  template<MemoryLocation DstLoc, typename T, MemoryLocation SrcLoc>
  auto to_buffer_async(const Buffer<T, SrcLoc>& src, cudaStream_t stream = cudaStreamDefault) {
    return to_buffer_async<DstLoc>(src.view(), stream);
  }

  template<typename T>
  auto to_host_pinned_buffer_async(const DeviceBuffer<T>& src, cudaStream_t stream = cudaStreamDefault) {
    return to_host_pinned_buffer_async(src.view(), stream);
  }

  template<typename T>
  auto to_device_buffer_async(const HostPinnedBuffer<T>& src, cudaStream_t stream = cudaStreamDefault) {
    return to_device_buffer_async(src.view(), stream);
  }

  template<typename T>
  auto to_device_buffer_async(const DeviceBuffer<T>& src, cudaStream_t stream = cudaStreamDefault) {
    return to_device_buffer_async(src.view(), stream);
  }
} // namespace gpu_lab
