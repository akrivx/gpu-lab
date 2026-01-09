#pragma once

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

#include "byte_storage.hpp"
#include "image_view.hpp"
#include "memory_location.hpp"
#include "memory_resource.hpp"
#include "pitched_element.hpp"
#include "tiled_image_view.hpp"

namespace gpu_lab {
  template <PitchedElement T, MemoryLocation Loc>
  class [[nodiscard]] Image {
    using resource_type = detail::DefaultMemoryResource<Loc>;

  public:
    using element_type = T;
    using view_type = ImageView<T, Loc>;
    using const_view_type = ImageView<const T, Loc>;

    static constexpr MemoryLocation location() { return Loc; }

    Image() noexcept = default;

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    Image(std::size_t h, std::size_t w)
        : storage_{w * sizeof(T), h, alignof(T)}
        , width_{w}
        , pitch_{storage_.stride_bytes() / sizeof(T)} {
      assert((pitch_bytes() % sizeof(T)) == 0); // PitchedElement guarantees that
      assert((width_bytes() / sizeof(T)) == w);
    }

    Image(Image&& o) noexcept
        : storage_{std::exchange(o.storage_, {})}
        , width_{std::exchange(o.width_, 0)}
        , pitch_{std::exchange(o.pitch_, 0)} {}

    Image& operator=(Image&& o) noexcept {
      if (this != &o) {
        storage_ = std::exchange(o.storage_, {});
        width_ = std::exchange(o.width_, 0);
        pitch_ = std::exchange(o.pitch_, 0);
      }
      return *this;
    }

    element_type* data() noexcept { return static_cast<T*>(storage_.data()); }
    const element_type* data() const noexcept { return static_cast<const T*>(storage_.data()); }
    const element_type* cdata() const noexcept { return static_cast<const T*>(storage_.data()); }

    std::size_t width() const noexcept { return width_; }
    std::size_t height() const noexcept { return storage_.block_count(); }
    std::size_t pitch() const noexcept { return pitch_; }

    std::size_t width_bytes() const noexcept { return storage_.block_bytes(); }
    std::size_t pitch_bytes() const noexcept { return storage_.stride_bytes(); }
    std::size_t size_bytes() const noexcept { return storage_.size_bytes(); }

    bool empty() const noexcept { return size_bytes() == 0; }

    view_type view() noexcept { return image_view<Loc>(data(), pitch(), height(), width()); }

    const_view_type view() const noexcept {
      return image_view<Loc>(data(), pitch(), height(), width());
    }

    const_view_type cview() const noexcept {
      return image_view<Loc>(data(), pitch(), height(), width());
    }

  private:
    detail::StridedByteStorage<resource_type> storage_ = {};
    std::size_t width_ = {};
    std::size_t pitch_ = {};
  };

  namespace detail {
    inline void validate_tiling_shape(std::size_t img_height,
                                      std::size_t img_width,
                                      std::size_t tile_height,
                                      std::size_t tile_width) {
      if (tile_height == 0 || tile_width == 0) {
        throw std::invalid_argument("tiled_image_view: Tile dimensions must be > 0.");
      }

      if (tile_height > img_height || tile_width > img_width) {
        throw std::invalid_argument(
            "tiled_image_view: Tile dimensions must not exceed image dimensions.");
      }

      if (img_height % tile_height != 0) {
        throw std::invalid_argument(
            "tiled_image_view: Image height is not divisible by tile height.");
      }

      if (img_width % tile_width != 0) {
        throw std::invalid_argument(
            "tiled_image_view: Image width is not divisible by tile width.");
      }
    }

    template <typename Img, typename TileExt>
    auto make_tiled_image_view(Img& img, const TileExt& tile_ext) {
      validate_tiling_shape(
          img.height(), img.width(), get_extent<0>(tile_ext), get_extent<1>(tile_ext));
      return ::gpu_lab::tiled_image_view(img.view(), tile_ext);
    }

    template <typename Img>
    auto make_tiled_image_view(Img& img, std::size_t tile_height, std::size_t tile_width) {
      return make_tiled_image_view(img, DynamicImageViewExtents{tile_height, tile_width});
    }
  } // namespace detail

  template <std::size_t TileH, std::size_t TileW, PitchedElement T, MemoryLocation Loc>
  auto tiled_image_view(const Image<T, Loc>& img,
                        const ImageViewExtents<TileH, TileW>& tile_ext = {}) {
    return detail::make_tiled_image_view(img, tile_ext);
  }

  template <std::size_t TileH, std::size_t TileW, PitchedElement T, MemoryLocation Loc>
  auto tiled_image_view(Image<T, Loc>& img, const ImageViewExtents<TileH, TileW>& tile_ext = {}) {
    return detail::make_tiled_image_view(img, tile_ext);
  }

  template <PitchedElement T, MemoryLocation Loc>
  auto tiled_image_view(const Image<T, Loc>& img, std::size_t tile_height, std::size_t tile_width) {
    return detail::make_tiled_image_view(img, tile_height, tile_width);
  }

  template <PitchedElement T, MemoryLocation Loc>
  auto tiled_image_view(Image<T, Loc>& img, std::size_t tile_height, std::size_t tile_width) {
    return detail::make_tiled_image_view(img, tile_height, tile_width);
  }

  template <PitchedElement T>
  using HostImage = Image<T, MemoryLocation::Host>;

  template <PitchedElement T>
  using HostPinnedImage = Image<T, MemoryLocation::HostPinned>;

  template <PitchedElement T>
  using DeviceImage = Image<T, MemoryLocation::Device>;
} // namespace gpu_lab
