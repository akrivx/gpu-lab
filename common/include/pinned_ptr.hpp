#pragma once

#include <memory>
#include <type_traits>
#include <cuda_runtime.h>
#include "cuda_check.hpp"


namespace gpu_lab {

  namespace detail {

    struct PinnedPtrDeleter {
      void operator()(void* host_ptr) const noexcept {
        cudaFreeHost(host_ptr);
      }
    };

  } // namespace detail

  template<typename T>
  using UniquePinnedPtr = std::unique_ptr<T, detail::PinnedPtrDeleter>;
  
  template<typename T>
    requires std::is_unbounded_array_v<T>
  auto make_unique_pinned_ptr(size_t n) {
    T* pinned_ptr = {};
    CUDA_CHECK(cudaMallocHost(&pinned_ptr, n * sizeof(T)));
    return UniquePinnedPtr<T>{pinned_ptr};
  }

}