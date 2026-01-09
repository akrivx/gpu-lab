#pragma once

#include <cstddef>
#include <utility>

#include "memory_resource_concepts.hpp"

namespace gpu_lab::detail {
  template <ByteResource Resource>
  class BufferStorage {
    struct Block {
      void* ptr = nullptr;
      std::size_t size_bytes = 0;
      std::size_t alignment = 0;
    };

  public:
    BufferStorage() noexcept = default;

    BufferStorage(std::size_t nbytes, std::size_t alignment)
        : block_{Resource::allocate_bytes(nbytes, alignment), nbytes, alignment} {}

    ~BufferStorage() noexcept { reset(); }

    BufferStorage(const BufferStorage&) = delete;
    BufferStorage& operator=(const BufferStorage&) = delete;

    BufferStorage(BufferStorage&& o) noexcept
        : block_{std::exchange(o.block_, {})} {}

    BufferStorage& operator=(BufferStorage&& o) noexcept {
      if (this != &o) {
        reset(std::exchange(o.block_, {}));
      }
      return *this;
    }

    const void* data() const noexcept { return block_.ptr; }
    void* data() noexcept { return block_.ptr; }
    std::size_t size_bytes() const noexcept { return block_.size_bytes; }

  private:
    void reset(Block new_block = {}) noexcept {
      if (block_.ptr) {
        Resource::deallocate_bytes(block_.ptr, block_.size_bytes, block_.alignment);
      }
      block_ = new_block;
    }

    Block block_ = {};
  };
} // namespace gpu_lab::detail
