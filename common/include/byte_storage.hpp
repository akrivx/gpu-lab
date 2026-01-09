#pragma once

#include <cstddef>
#include <utility>

#include "byte_storage_traits.hpp"
#include "memory_resource_concepts.hpp"

namespace gpu_lab::detail {
  template <typename Traits>
  class BasicByteStorage {
  public:
    using allocation_type = typename Traits::allocation_type;

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

    const allocation_type& allocation() const noexcept { return alloc_; }

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
  using StridedByteStorage = BasicByteStorage<StridedBytesStorageTraits<Resource>>;
} // namespace gpu_lab::detail
