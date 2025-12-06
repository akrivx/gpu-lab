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
  } // namespace detail

  template<std::size_t H = cuda::std::dynamic_extent,
           std::size_t W = cuda::std::dynamic_extent>
  using ImageViewExtents = cuda::std::extents<std::size_t, H, W>;

  using DynamicImageViewExtents = ImageViewExtents<>;

  using ImageExtentRange = cuda::std::pair<std::size_t, std::size_t>;

  template<typename T,
           MemoryLocation Loc,
           typename Extents = DynamicImageViewExtents>
  using ImageView = cuda::std::mdspan<
    T,
    Extents,
    cuda::std::layout_stride,
    detail::ByteOffsetAccessor<T, Loc>
  >;

  template<MemoryLocation Loc,
           std::size_t H,
           std::size_t W,
           typename T>
  __host__ __device__ auto image_view(
    T*                     data,
    ImageViewExtents<H, W> extents,
    std::size_t            pitch_bytes)
  {
    cuda::std::array<std::size_t, 2> strides{pitch_bytes, sizeof(T)};
    auto mapping = cuda::std::layout_stride::mapping(extents, strides);
    return ImageView<T, Loc>{data, mapping};
  }

  template<MemoryLocation Loc, typename T>
  __host__ __device__ auto image_view(
    T*          data,
    std::size_t width,
    std::size_t height,
    std::size_t pitch_bytes)
  {
    return image_view<Loc>(data, DynamicImageViewExtents{height, width}, pitch_bytes);
  }

  namespace detail {
    template<typename T, MemoryLocation Loc, typename Extents, typename R, typename C>
    __host__ __device__ ImageView<T, Loc, Extents> subview(ImageView<T, Loc, Extents> v, R r, C c) {
      return cuda::std::submdspan(v, r, c);
    }   
  } // namespace detail

  template<typename T, MemoryLocation Loc, typename Extents>
  __host__ __device__ auto subrows(ImageView<T, Loc, Extents> v, ImageExtentRange r) {
    return detail::subview(v, r, cuda::std::full_extent);
  }

  template<typename T, MemoryLocation Loc, typename Extents>
  __host__ __device__ auto subcols(ImageView<T, Loc, Extents> v, ImageExtentRange c) {
    return detail::subview(v, cuda::std::full_extent, c);
  }

  template<typename T, MemoryLocation Loc, typename Extents>
  __host__ __device__ auto subview(ImageView<T, Loc, Extents> v, ImageExtentRange r, ImageExtentRange c) {
    return detail::subview(v, r, c);
  }

  template<typename T, MemoryLocation Loc, typename Extents>
  __host__ __device__ auto as_const(ImageView<T, Loc, Extents> v) {
    using value_type = typename ImageView<T, Loc, Extents>::value_type;
    return ImageView<const value_type, Loc, Extents>{v.data_handle(), v.mapping()};
  }

  template<typename T, typename Extents = DynamicImageViewExtents>
  using HostImageView = ImageView<T, MemoryLocation::Host, Extents>;

  template<typename T, typename Extents = DynamicImageViewExtents>
  using HostPinnedImageView = ImageView<T, MemoryLocation::HostPinned, Extents>;

  template<typename T, typename Extents = DynamicImageViewExtents>
  using DeviceImageView = ImageView<T, MemoryLocation::Device, Extents>;
} // namespace gpu_lab
