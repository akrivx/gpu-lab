#pragma once

#include <cassert>
#include <cstddef>

#include <cuda/std/array>
#include <cuda/std/mdspan>

#include <cuda_runtime.h>

#include "image_view.hpp"
#include "memory_location.hpp"
#include "pitched_element.hpp"

namespace gpu_lab {
  namespace detail {
    template<typename T, MemoryLocation Loc, typename TileExtents>
    struct TileAccessor {
      using tile_view_type   = ImageView<T, Loc, TileExtents>;
      using element_type     = tile_view_type;
      using reference        = element_type;      // return tile view by value
      using data_handle_type = T*;
      using offset_policy    = TileAccessor<T, Loc, TileExtents>;
      using index_type       = std::ptrdiff_t;

      cuda::std::array<std::size_t, 2>  strides_;
      [[no_unique_address]] TileExtents tile_extents_{};

      static constexpr MemoryLocation location() { return Loc; }

      __host__ __device__
      constexpr TileAccessor() noexcept = default;

      __host__ __device__
      constexpr TileAccessor(
        const cuda::std::array<std::size_t, 2>& strides,
        const TileExtents& tile_extents) noexcept
        : strides_{strides}
        , tile_extents_{tile_extents}
      {}

      __host__ __device__
      constexpr reference access(data_handle_type p, index_type i) const noexcept
      {
        // i is the index of the tile's top-left pixel
        auto tile_base = offset(p, i);
        return image_view<Loc>(tile_base, strides_, tile_extents_);
      }

      __host__ __device__
      constexpr data_handle_type offset(data_handle_type p, index_type i) const noexcept
      {
        return p + i;
      }
    };

    template<
      std::size_t R,
      std::size_t H,
      std::size_t W>
    __host__ __device__
    constexpr std::size_t get_extent(const ImageViewExtents<H, W>& extents)
    {
      static_assert(R < 2);
    
      if constexpr (R == 0) {
        if constexpr (H != cuda::std::dynamic_extent) {
          return H;                      // static height
        }
        else {
          return extents.extent(0);        // dynamic height
        }
      } else {
        if constexpr (W != cuda::std::dynamic_extent) {
          return W;                      // static width
        }
        else {
          return extents.extent(1);        // dynamic width
        }
      }
    }

    template<std::size_t TileH, std::size_t TileW>
    DynamicImageViewExtents tile_grid_extents(
      DynamicImageViewExtents        image,
      ImageViewExtents<TileH, TileW> tile)
    {
      assert((image.extent(0) % get_extent<0>(tile)) == 0);
      assert((image.extent(1) % get_extent<1>(tile)) == 0);

      const std::size_t grid_h = image.extent(0) / get_extent<0>(tile);
      const std::size_t grid_w = image.extent(1) / get_extent<1>(tile);

      return DynamicImageViewExtents{grid_h, grid_w};
    }
  } // namespace detail

  template<
    PitchedElement T,
    MemoryLocation Loc,
    typename TileExtents = DynamicImageViewExtents>
  using TiledImageView = cuda::std::mdspan<
    ImageView<T, Loc, TileExtents>, // element_type = tile view
    DynamicImageViewExtents,        // grid extents (currently dynamic)
    cuda::std::layout_stride,
    detail::TileAccessor<T, Loc, TileExtents>>;

    template<
    MemoryLocation Loc,
    std::size_t TileH,
    std::size_t TileW,
    PitchedElement T>
  __host__ __device__ auto tiled_image_view(
    T*                                      data,
    const cuda::std::array<std::size_t, 2>& image_strides,
    const DynamicImageViewExtents&          image_extents,
    const ImageViewExtents<TileH, TileW>&   tile_extents = {})
  {
    using detail::get_extent;
    using detail::tile_grid_extents;
    using tile_extents_t = ImageViewExtents<TileH, TileW>;
    using accessor_t = detail::TileAccessor<T, Loc, tile_extents_t>;

    const auto tile_h = get_extent<0>(tile_extents);
    const auto tile_w = get_extent<1>(tile_extents);

    assert((image_extents.extent(0) % tile_h) == 0);
    assert((image_extents.extent(1) % tile_w) == 0);

    // strides to jump from one tile to the next
    cuda::std::array<std::size_t, 2> tile_grid_strides{
      image_strides[0] * tile_h,
      image_strides[1] * tile_w};
    
    auto mapping = cuda::std::layout_stride::mapping(
      tile_grid_extents(image_extents, tile_extents),
      tile_grid_strides);
  
    accessor_t accessor{image_strides, tile_extents};
    return TiledImageView<T, Loc, tile_extents_t>{data, mapping, accessor};
  }

  template<
    MemoryLocation Loc,
    std::size_t TileH,
    std::size_t TileW,
    PitchedElement T>
  __host__ __device__ auto tiled_image_view(
    T*                                    data,
    std::size_t                           image_pitch,
    const DynamicImageViewExtents&        image_extents,
    const ImageViewExtents<TileH, TileW>& tile_extents = {})
  {
    return tiled_image_view<Loc>(
      data,
      {image_pitch, 1},
      image_extents,
      tile_extents);
  }

  template<
    MemoryLocation Loc,
    std::size_t TileH,
    std::size_t TileW,
    PitchedElement T>
  __host__ __device__ auto tiled_image_view(
    T*                                    data,
    std::size_t                           image_pitch,
    std::size_t                           image_height,
    std::size_t                           image_width,
    const ImageViewExtents<TileH, TileW>& tile_extents = {})
  {
    return tiled_image_view<Loc>(
      data,
      {image_pitch, 1},
      {image_height, image_width},
      tile_extents);
  }

  template<
    std::size_t TileH,
    std::size_t TileW,
    PitchedElement T,
    MemoryLocation Loc>
  __host__ __device__ auto tiled_image_view(
    ImageView<T, Loc>                     v,
    const ImageViewExtents<TileH, TileW>& tile_extents = {})
  {
    return tiled_image_view<Loc>(
      &v(0, 0),
      {v.stride(0), v.stride(1)},
      v.extents(),
      tile_extents);
  }

  template<PitchedElement T, MemoryLocation Loc>
  __host__ __device__ auto tiled_image_view(
    ImageView<T, Loc> v,
    std::size_t       tile_height,
    std::size_t       tile_width)
  {
    return tiled_image_view(
      v,
      DynamicImageViewExtents{tile_height, tile_width});
  }

  template<PitchedElement T, typename TileExtents = DynamicImageViewExtents>
  using HostTiledImageView = TiledImageView<T, MemoryLocation::Host, TileExtents>;

  template<PitchedElement T, typename TileExtents = DynamicImageViewExtents>
  using HostPinnedTiledImageView = TiledImageView<T, MemoryLocation::HostPinned, TileExtents>;

  template<PitchedElement T, typename TileExtents = DynamicImageViewExtents>
  using DeviceTiledImageView = TiledImageView<T, MemoryLocation::Device, TileExtents>;

} // namespace gpu_lab
