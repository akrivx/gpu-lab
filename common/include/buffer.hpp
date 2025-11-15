#pragma once

#include <cstddef>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <span>
#include "device_ptr.hpp"
#include "pinned_ptr.hpp"


namespace gpu_lab {

  enum class BufferLocation { HOST_PAGEABLE, HOST_PINNED, DEVICE };

  namespace detail {

    template<typename T, BufferLocation loc>
    struct BufferTraits {
      using element_type = T;
      using pointer_handle = std::conditional_t<
        loc == BufferLocation::HOST_PAGEABLE,
        std::unique_ptr<T[]>,
        std::conditional_t<
          loc == BufferLocation::HOST_PINNED,
          UniquePinnedPtr<T[]>,
          UniqueDevicePtr<T[]>
        >
      >;

      static auto allocate(size_t size) {
        if constexpr (loc == BufferLocation::HOST_PAGEABLE) {
          return std::make_unique_for_overwrite<T[]>(size);
        }
        else if constexpr (loc == BufferLocation::HOST_PINNED) {
          return make_unique_pinned_ptr<T[]>(size);
        }
        else { // BufferLocation::DEVICE
          return make_unique_device_ptr<T[]>(size);
        }
      }
    };

  } // namespace detail

  template<typename T, BufferLocation Loc>
  class Buffer {
    using traits = detail::BufferTraits<T, Loc>;
    
  public:
    using pointer_handle = typename traits::pointer_handle;
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

    Buffer(pointer_handle&& ptr, size_t size)
      : ptr_{std::move(ptr)}
      , size_{size}
    {}

    pointer_handle release() {
      auto ptr = std::exchange(ptr_, {});
      size_ = {};
      return std::move(ptr);
    }

    T* data() noexcept { return ptr_.get(); }
    const T* data() const noexcept { return ptr_.get(); }

    size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

  private:
    pointer_handle ptr_ = {};
    size_t         size_ = {};
  };

  template<typename T, BufferLocation Loc>
  struct BufferView {
    using element_type = T;

    BufferView(Buffer<std::remove_const_t<T>, Loc>& buf) noexcept
      requires (!std::is_const_v<T>)
      : data_{buf.data()}
      , size_{buf.size()}
    {}

    BufferView(const Buffer<std::remove_const_t<T>, Loc>& buf) noexcept
      requires (std::is_const_v<T>)
      : data_{buf.data()}
      , size_{buf.size()}
    {}
  
    BufferView(T* data, size_t size) noexcept
      : data_{data}
      , size_{size}
    {}

    T* data() const noexcept { return data_; }
    size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    BufferView subspan(size_t offset, size_t count) const noexcept {
      assert((offset + count) <= size_);
      return {data_ + offset, count};
    }

    BufferView take(size_t count) const noexcept {
      assert(count <= size_);
      return subspan(0, count);
    }

    BufferView drop(size_t count) const noexcept {
      assert(count <= size_);
      return subspan(count, size_ - count);
    }

    T*     data_ = {};
    size_t size_ = {};
  };

  template<typename T, BufferLocation Loc>
  auto view(const Buffer<T, Loc>& buf) noexcept {
    return BufferView<const T, Loc>{buf};
  }

  template<typename T, BufferLocation Loc>
  auto view(Buffer<T, Loc>& buf) noexcept {
    return BufferView<T, Loc>{buf};
  }

  namespace detail {

    template<BufferLocation SrcLoc, BufferLocation DstLoc>
    constexpr auto get_memcpy_kind() {
      if constexpr (SrcLoc == BufferLocation::DEVICE) {
        if constexpr (DstLoc == BufferLocation::DEVICE) {
          return cudaMemcpyDeviceToDevice;
        }
        else {
          return cudaMemcpyDeviceToHost;
        }
      }
      else if constexpr (DstLoc == BufferLocation::DEVICE) {
        return cudaMemcpyHostToDevice;
      }
      else {
        return cudaMemcpyHostToHost;
      }
    }

  }

  template<typename T, BufferLocation SrcLoc, BufferLocation DstLoc>
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

  template<typename T, BufferLocation SrcLoc, BufferLocation DstLoc>
    requires (SrcLoc == BufferLocation::DEVICE || DstLoc == BufferLocation::DEVICE)
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
  using DeviceBuffer = Buffer<T, BufferLocation::DEVICE>;

  template<typename T>
  using HostPinnedBuffer = Buffer<T, BufferLocation::HOST_PINNED>;

  template<typename T>
  using HostPageableBuffer = Buffer<T, BufferLocation::HOST_PAGEABLE>;

}