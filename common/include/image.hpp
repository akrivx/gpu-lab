#pragma once

#include <array>
#include <memory>
#include <utility>

#include <cuda_runtime.h>

#include "memory_location.hpp"
#include "unique_array.hpp"
#include "image_view.hpp"

namespace gpu_lab {
  template<typename T, MemoryLocation Loc>
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

    Image(size_t w, size_t h)
      : width_{w}
      , height_{h}
    {
      std::tie(handle_, pitch_bytes_) = make_unique_array2d<T, Loc>(w, h);
    }

    Image(Image&& o) noexcept
      : width_{std::exchange(o.width_, {})}
      , height_{std::exchange(o.height_, {})}
      , pitch_bytes_{std::exchange(o.pitch_bytes_, {})}
      , handle_{std::exchange(o.handle_, {})}
    {}

    Image& operator=(Image&& o) noexcept {
      if (this != &o) {
        width_ = std::exchange(o.width_, {});
        height_ = std::exchange(o.height_, {});
        pitch_bytes_ = std::exchange(o.pitch_bytes_, {});
        handle_ = std::exchange(o.handle_, {});
      }
      return *this;
    }

    handle_type release() noexcept {
      width_ = {};
      height_ = {};
      pitch_bytes_ = {};
      return std::exchange(handle_, {});
    }

    element_type* data() noexcept { return handle_.get(); }
    const element_type* data() const noexcept { return handle_.get(); }
    const element_type* cdata() const noexcept { return handle_.get(); }

    size_t width() const noexcept { return width_; }
    size_t height() const noexcept { return height_; }
    size_t pitch_bytes() const noexcept { return pitch_bytes_; }
    size_t size_bytes() const noexcept { return height_ * pitch_bytes_; }
    bool empty() const noexcept { return handle_ == nullptr; }

    view_type view() noexcept { return image_view(data(), width_, height_, pitch_bytes_); }
    const_view_type view() const noexcept { return image_view(data(), width_, height_, pitch_bytes_); }
    const_view_type cview() const noexcept { return image_view(data(), width_, height_, pitch_bytes_); }
    
  private:
    size_t width_       = {};
    size_t height_      = {};
    size_t pitch_bytes_ = {};
    handle_type handle_ = {};
  };
}
