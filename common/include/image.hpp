#pragma once

#include <cassert>
#include <cstddef>
#include <utility>
#include <stdexcept>

#include <cuda_runtime.h>

#include "image_view.hpp"
#include "memory_location.hpp"
#include "pitched_element.hpp"
#include "tiled_image_view.hpp"
#include "unique_array.hpp"

namespace gpu_lab {
  template<PitchedElement T, MemoryLocation Loc>
  class [[nodiscard]] Image {
  public:
    using handle_type = UniqueArray<T, Loc>;
    using element_type = T;
    using view_type = ImageView<T, Loc>;
    using const_view_type = ImageView<const T, Loc>;

    static constexpr MemoryLocation location() { return Loc; }

    Image() noexcept = default;

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    Image(std::size_t w, std::size_t h)
      : width_{w}
      , height_{h}
    {
      auto [handle, pitch_bytes] = make_unique_array2d<T, Loc>(w, h);
      assert((pitch_bytes % sizeof(element_type)) == 0); // PitchedElement guarantees that
      handle_ = std::move(handle);
      pitch_ = pitch_bytes / sizeof(element_type);
    }

    Image(Image&& o) noexcept
      : width_{std::exchange(o.width_, {})}
      , height_{std::exchange(o.height_, {})}
      , pitch_{std::exchange(o.pitch_, {})}
      , handle_{std::exchange(o.handle_, {})}
    {}

    Image& operator=(Image&& o) noexcept {
      if (this != &o) {
        width_ = std::exchange(o.width_, {});
        height_ = std::exchange(o.height_, {});
        pitch_ = std::exchange(o.pitch_, {});
        handle_ = std::exchange(o.handle_, {});
      }
      return *this;
    }

    handle_type release() noexcept {
      width_ = {};
      height_ = {};
      pitch_ = {};
      return std::exchange(handle_, {});
    }

    element_type* data() noexcept { return handle_.get(); }
    const element_type* data() const noexcept { return handle_.get(); }
    const element_type* cdata() const noexcept { return handle_.get(); }

    std::size_t width() const noexcept { return width_; }
    std::size_t height() const noexcept { return height_; }
    std::size_t pitch() const noexcept { return pitch_; }
    std::size_t pitch_bytes() const noexcept { return pitch_ * sizeof(element_type); }
    std::size_t size_bytes() const noexcept { return height_ * pitch_bytes(); }
    bool empty() const noexcept { return handle_ == nullptr; }

    view_type view() noexcept { return image_view(data(), width_, height_, pitch_); }
    const_view_type view() const noexcept { return image_view(data(), width_, height_, pitch_); }
    const_view_type cview() const noexcept { return image_view(data(), width_, height_, pitch_); }
    
  private:
    std::size_t width_  = {};
    std::size_t height_ = {};
    std::size_t pitch_  = {}; // in elements
    handle_type handle_ = {};
  };

  namespace detail {
    inline void validate_tiling_shape(
      std::size_t img_height,
      std::size_t img_width,
      std::size_t tile_height,
      std::size_t tile_width)
    {
      if (tile_height == 0 || tile_width == 0) {
        throw std::invalid_argument("Tile dimensions must be > 0.");
      }
  
      if (tile_height > img_height || tile_width > img_width) {
        throw std::invalid_argument("Tile dimensions must not exceed image dimensions.");
      }
  
      if (img_height % tile_height != 0) {
        throw std::invalid_argument("Image height is not divisible by tile height.");
      }
  
      if (img_width % tile_width != 0) {
        throw std::invalid_argument("Image width is not divisible by tile width.");
      }
    }

    template<typename Img, typename TileExt>
    auto make_tiled_image_view(Img& img, const TileExt& tile_ext) {
      validate_tiling_shape(
        img.height(),
        img.width(),
        get_extent<0>(tile_ext),
        get_extent<1>(tile_ext));
      return ::gpu_lab::tiled_image_view(img.view(), tile_ext);
    }

    template<typename Img>
    auto make_tiled_image_view(Img& img, std::size_t tile_width, std::size_t tile_height) {
      return make_tiled_image_view(img, DynamicImageViewExtents{tile_height, tile_width});
    }
  } // namespace detail

  template<
    std::size_t TileH,
    std::size_t TileW,
    PitchedElement T,
    MemoryLocation Loc>
  auto tiled_image_view(
    const Image<T, Loc>& img,
    const ImageViewExtents<TileH, TileW>& tile_ext = {})
  {
    return detail::make_tiled_image_view(img, tile_ext);
  }

  template<
    std::size_t TileH,
    std::size_t TileW,
    PitchedElement T,
    MemoryLocation Loc>
  auto tiled_image_view(
    Image<T, Loc>& img,
    const ImageViewExtents<TileH, TileW>& tile_ext = {})
  {
    return detail::make_tiled_image_view(img, tile_ext);
  }

  template<PitchedElement T, MemoryLocation Loc>
  auto tiled_image_view(
    const Image<T, Loc>& img,
    std::size_t          tile_width,
    std::size_t          tile_height)
  {
    return detail::make_tiled_image_view(img, tile_width, tile_height);
  }

  template<PitchedElement T, MemoryLocation Loc>
  auto tiled_image_view(
    Image<T, Loc>& img,
    std::size_t    tile_width,
    std::size_t    tile_height)
  {
    return detail::make_tiled_image_view(img, tile_width, tile_height);
  }
} // namespace gpu_lab
