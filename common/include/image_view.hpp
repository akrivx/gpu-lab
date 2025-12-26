#pragma once

#include <cstddef>
#include <cuda/std/array>
#include <cuda/std/mdspan>

#include <cuda_runtime.h>

#include "memory_location.hpp"
#include "pitched_element.hpp"

namespace gpu_lab {
  namespace detail {
    template <class T, MemoryLocation Loc>
    struct ImageAccessor : cuda::std::default_accessor<T> {
      // Base typedefs inherited but offset_policy must be redefined
      using offset_policy = ImageAccessor;

      using base_type = cuda::std::default_accessor<T>;

      // Inherit constructors to use like default_accessor
      using base_type::base_type;

      static constexpr MemoryLocation location() noexcept { return Loc; }
    };
  } // namespace detail

  template <std::size_t H = cuda::std::dynamic_extent, std::size_t W = cuda::std::dynamic_extent>
  using ImageViewExtents = cuda::std::extents<std::size_t, H, W>;

  using DynamicImageViewExtents = ImageViewExtents<>;

  using ImageExtentRange = cuda::std::pair<std::size_t, std::size_t>;

  template <PitchedElement T, MemoryLocation Loc, typename Extents = DynamicImageViewExtents>
  using ImageView =
      cuda::std::mdspan<T, Extents, cuda::std::layout_stride, detail::ImageAccessor<T, Loc>>;

  template <MemoryLocation Loc, std::size_t H, std::size_t W, PitchedElement T>
  __host__ __device__ auto image_view(T* data,
                                      const cuda::std::array<std::size_t, 2>& strides,
                                      const ImageViewExtents<H, W>& extents = {}) {
    auto mapping = cuda::std::layout_stride::mapping(extents, strides);
    return ImageView<T, Loc, ImageViewExtents<H, W>>{data, mapping};
  }

  template <MemoryLocation Loc, std::size_t H, std::size_t W, PitchedElement T>
  __host__ __device__ auto
  image_view(T* data, std::size_t pitch, const ImageViewExtents<H, W>& extents = {}) {
    return image_view<Loc>(data, {pitch, 1}, extents);
  }

  template <MemoryLocation Loc, PitchedElement T>
  __host__ __device__ auto
  image_view(T* data, std::size_t pitch, std::size_t height, std::size_t width) {
    return image_view<Loc>(data, pitch, DynamicImageViewExtents{height, width});
  }

  namespace detail {
    template <PitchedElement T, MemoryLocation Loc, typename Extents, typename R, typename C>
    __host__ __device__ ImageView<T, Loc, Extents> subview(ImageView<T, Loc, Extents> v, R r, C c) {
      return cuda::std::submdspan(v, r, c);
    }
  } // namespace detail

  template <PitchedElement T, MemoryLocation Loc, typename Extents>
  __host__ __device__ auto subrows(ImageView<T, Loc, Extents> v, ImageExtentRange r) {
    return detail::subview(v, r, cuda::std::full_extent);
  }

  template <PitchedElement T, MemoryLocation Loc, typename Extents>
  __host__ __device__ auto subcols(ImageView<T, Loc, Extents> v, ImageExtentRange c) {
    return detail::subview(v, cuda::std::full_extent, c);
  }

  template <PitchedElement T, MemoryLocation Loc, typename Extents>
  __host__ __device__ auto
  subview(ImageView<T, Loc, Extents> v, ImageExtentRange r, ImageExtentRange c) {
    return detail::subview(v, r, c);
  }

  template <PitchedElement T, MemoryLocation Loc, typename Extents>
  __host__ __device__ auto as_const(ImageView<T, Loc, Extents> v) {
    using value_type = typename ImageView<T, Loc, Extents>::value_type;
    return ImageView<const value_type, Loc, Extents>{v.data_handle(), v.mapping()};
  }

  template <PitchedElement T, typename Extents = DynamicImageViewExtents>
  using HostImageView = ImageView<T, MemoryLocation::Host, Extents>;

  template <PitchedElement T, typename Extents = DynamicImageViewExtents>
  using HostPinnedImageView = ImageView<T, MemoryLocation::HostPinned, Extents>;

  template <PitchedElement T, typename Extents = DynamicImageViewExtents>
  using DeviceImageView = ImageView<T, MemoryLocation::Device, Extents>;
} // namespace gpu_lab
