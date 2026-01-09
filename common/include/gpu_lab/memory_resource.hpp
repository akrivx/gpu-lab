#pragma once

#include <cstddef>

#include "memory_location.hpp"
#include "strided_bytes.hpp"

namespace gpu_lab::detail {
  struct HostMemoryResource {
    static constexpr MemoryLocation location = MemoryLocation::Host;

    static void* allocate_bytes(std::size_t nbytes, std::size_t alignment);

    static StridedBytes allocate_strided_bytes(std::size_t block_bytes,
                                               std::size_t block_count,
                                               std::size_t block_alignment);

    static void deallocate_bytes(void* ptr,
                                 [[maybe_unused]] std::size_t nbytes,
                                 std::size_t alignment) noexcept;
  };

  struct HostPinnedMemoryResource {
    static constexpr MemoryLocation location = MemoryLocation::HostPinned;

    static void* allocate_bytes(std::size_t nbytes, [[maybe_unused]] std::size_t alignment);

    static StridedBytes allocate_strided_bytes(std::size_t block_bytes,
                                               std::size_t block_count,
                                               std::size_t block_alignment);

    static void deallocate_bytes(void* ptr,
                                 [[maybe_unused]] std::size_t nbytes,
                                 [[maybe_unused]] std::size_t alignment) noexcept;
  };

  struct DeviceMemoryResource {
    static constexpr MemoryLocation location = MemoryLocation::Device;

    static void* allocate_bytes(std::size_t nbytes, [[maybe_unused]] std::size_t alignment);

    static StridedBytes allocate_strided_bytes(std::size_t block_bytes,
                                               std::size_t block_count,
                                               std::size_t block_alignment);

    static void deallocate_bytes(void* ptr,
                                 [[maybe_unused]] std::size_t nbytes,
                                 [[maybe_unused]] std::size_t alignment) noexcept;
  };

  template <MemoryLocation Loc>
  struct DefaultMemoryResourceFor;

  template <>
  struct DefaultMemoryResourceFor<MemoryLocation::Host> {
    using type = HostMemoryResource;
  };

  template <>
  struct DefaultMemoryResourceFor<MemoryLocation::HostPinned> {
    using type = HostPinnedMemoryResource;
  };

  template <>
  struct DefaultMemoryResourceFor<MemoryLocation::Device> {
    using type = DeviceMemoryResource;
  };

  template <MemoryLocation Loc>
  using DefaultMemoryResource = DefaultMemoryResourceFor<Loc>::type;
} // namespace gpu_lab::detail
