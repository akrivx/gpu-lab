#pragma once

#include <cstddef>
#include <utility>

#include "gpu_lab/memory_location.hpp"

#include "gpu_lab/detail/byte_storage_traits.hpp"
#include "gpu_lab/detail/memory_resource.hpp"
#include "gpu_lab/detail/memory_resource_concepts.hpp"

namespace gpu_lab::detail {
  template <typename Traits>
  class BasicByteStorage {
    using allocation_type = typename Traits::allocation_type;

  public:
    BasicByteStorage() noexcept = default;

    template <typename... Args>
    explicit BasicByteStorage(Args&&... args)
        : alloc_{Traits::allocate(std::forward<Args>(args)...)} {}

    ~BasicByteStorage() noexcept { reset(); }

    BasicByteStorage(const BasicByteStorage&) = delete;
    BasicByteStorage& operator=(const BasicByteStorage&) = delete;

    BasicByteStorage(BasicByteStorage&& o) noexcept
        : alloc_{std::exchange(o.alloc_, {})} {}

    BasicByteStorage& operator=(BasicByteStorage&& o) noexcept {
      if (this != &o) {
        reset(std::exchange(o.alloc_, {}));
      }
      return *this;
    }

    const void* data() const noexcept { return alloc_.ptr; }
    void* data() noexcept { return alloc_.ptr; }

    std::size_t size_bytes() const noexcept { return Traits::size_bytes(alloc_); }

    std::size_t stride_bytes() const noexcept
      requires(Traits::is_strided)
    {
      return Traits::stride_bytes(alloc_);
    }

    std::size_t block_bytes() const noexcept
      requires(Traits::is_strided)
    {
      return Traits::block_bytes(alloc_);
    }

    std::size_t block_count() const noexcept
      requires(Traits::is_strided)
    {
      return Traits::block_count(alloc_);
    }

  private:
    void reset(allocation_type new_alloc = {}) noexcept {
      Traits::deallocate(alloc_);
      alloc_ = new_alloc;
    }

  private:
    allocation_type alloc_ = {};
  };

  template <ByteResource Resource>
  using ByteStorage = BasicByteStorage<ByteStorageTraits<Resource>>;

  template <StridedByteResource Resource>
  using StridedByteStorage = BasicByteStorage<StridedByteStorageTraits<Resource>>;
  
  template <MemoryLocation Loc>
  using DefaultByteStorage = ByteStorage<DefaultMemoryResource<Loc>>;

  template <MemoryLocation Loc>
  using DefaultStridedByteStorage = StridedByteStorage<DefaultMemoryResource<Loc>>;
} // namespace gpu_lab::detail
