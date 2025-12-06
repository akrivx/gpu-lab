#pragma once

#include <cstddef>
#include <cuda/std/mdspan>

#include <cuda_runtime.h>

#include "memory_location.hpp"

namespace gpu_lab {
  namespace detail {
    template<typename T, MemoryLocation Loc>
    struct ByteOffsetAccessor {
      using element_type     = T;
      using reference        = element_type&;
      using data_handle_type = element_type*;
      using offset_policy    = ByteOffsetAccessor<T, Loc>;
      using index_type       = std::ptrdiff_t;

      static constexpr MemoryLocation location() { return Loc; }

      __host__ __device__
      constexpr reference access(data_handle_type p, index_type i) const noexcept {
        return *offset(p, i);
      }

      __host__ __device__
      constexpr data_handle_type offset(data_handle_type p, index_type i) const noexcept {
        // i is a linear offset IN BYTES
        using byte_t = std::conditional_t<std::is_const_v<element_type>, const std::byte, std::byte>;
        auto bp = reinterpret_cast<byte_t*>(p);
        return reinterpret_cast<data_handle_type>(bp + i);
      }
    };

    using ImageViewExtents = cuda::std::extents<
      std::size_t,
      cuda::std::dynamic_extent,
      cuda::std::dynamic_extent
    >;
  } // namespace detail

  using ImageExtentRange = cuda::std::pair<std::size_t, std::size_t>;

  template<typename T, MemoryLocation Loc>
  using ImageView = cuda::std::mdspan<
    T,
    detail::ImageViewExtents,
    cuda::std::layout_stride,
    detail::ByteOffsetAccessor<T, Loc>
  >;

  template<MemoryLocation Loc, typename T>
  __host__ __device__ auto image_view(
    T*          data,
    std::size_t width,
    std::size_t height,
    std::size_t pitch_bytes)
  {
    detail::ImageViewExtents extents{height, width};
    cuda::std::array<std::size_t, 2> strides{pitch_bytes, sizeof(T)};
    auto mapping = cuda::std::layout_stride::mapping(extents, strides);
    return ImageView<T, Loc>{data, mapping};
  }

  namespace detail {
    template<typename T, MemoryLocation Loc, typename R, typename C>
    __host__ __device__ ImageView<T, Loc> subview(ImageView<T, Loc> v, R r, C c) {
      return cuda::std::submdspan(v, r, c);
    }   
  }

  template<typename T, MemoryLocation Loc>
  __host__ __device__ auto subrows(ImageView<T, Loc> v, ImageExtentRange r) {
    return detail::subview(v, r, cuda::std::full_extent);
  }

  template<typename T, MemoryLocation Loc>
  __host__ __device__ auto subcols(ImageView<T, Loc> v, ImageExtentRange c) {
    return detail::subview(v, cuda::std::full_extent, c);
  }

  template<typename T, MemoryLocation Loc>
  __host__ __device__ auto subview(ImageView<T, Loc> v, ImageExtentRange r, ImageExtentRange c) {
    return detail::subview(v, r, c);
  }

  template<typename T, MemoryLocation Loc>
  __host__ __device__ auto as_const(ImageView<T, Loc> v) {
    using value_type = typename ImageView<T, Loc>::value_type;
    return ImageView<const value_type, Loc>{v.data_handle(), v.mapping()};
  }

  template<typename T>
  using HostImageView = ImageView<T, MemoryLocation::Host>;

  template<typename T>
  using HostPinnedImageView = ImageView<T, MemoryLocation::HostPinned>;

  template<typename T>
  using DeviceImageView = ImageView<T, MemoryLocation::Device>;
} // namespace gpu_lab
