#pragma once

#include <concepts>
#include <cstddef>

#include "strided_bytes.hpp"

namespace gpu_lab::detail {
  template <typename R>
  concept ByteDeallocator = requires(void* p, std::size_t n, std::size_t a) {
    { R::deallocate_bytes(p, n, a) } noexcept;
  };

  template <typename R>
  concept ByteResource = ByteDeallocator<R> && requires(std::size_t n, std::size_t a) {
    { R::allocate_bytes(n, a) } -> std::same_as<void*>;
  };

  template <typename R>
  concept StridedByteResource =
      ByteDeallocator<R> && requires(std::size_t block, std::size_t count, std::size_t align) {
        { R::allocate_strided_bytes(block, count, align) } -> std::same_as<StridedBytes>;
      };
} // namespace gpu_lab::detail
