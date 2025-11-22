#pragma once

#include <memory>
#include <type_traits>
#include <cuda_runtime.h>
#include "cuda_check.hpp"

namespace gpu_lab {

  namespace detail {

    struct DevicePtrDeleter {
      void operator()(void* dev_ptr) const noexcept {
        cudaFree(dev_ptr);
      }
    };

  } // namespace detail

  template<typename T>
  using UniqueDevicePtr = std::unique_ptr<T, detail::DevicePtrDeleter>;

  template<typename T>
    requires std::is_unbounded_array_v<T>
  auto make_unique_device_ptr(size_t n) {
    using U = std::remove_extent_t<T>;
    U* dev_ptr = {};
    CUDA_CHECK(cudaMalloc(&dev_ptr, n * sizeof(U)));
    return UniqueDevicePtr<T>{dev_ptr};
  }

}