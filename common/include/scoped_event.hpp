#pragma once

#include <memory>

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace gpu_lab {
  namespace detail {
    struct EventDeleter {
      using pointer = cudaEvent_t;
      void operator()(cudaEvent_t e) const noexcept {
        cudaEventDestroy(e);
      }
    };

    using UniqueEvent = std::unique_ptr<cudaEvent_t, detail::EventDeleter>;

    inline auto make_unique_event(unsigned int flags) {
      cudaEvent_t e{};
      CUDA_CHECK(cudaEventCreateWithFlags(&e, flags));
      return UniqueEvent{e};
    }
  } // namespace detail

  class ScopedEvent {
  public:
    explicit ScopedEvent(unsigned int flags = cudaEventDefault)
      : ev_{detail::make_unique_event(flags)} {}

    void record(cudaStream_t stream = cudaStreamDefault) const {
      CUDA_CHECK(cudaEventRecord(ev_.get(), stream));
    }

    void sync() const {
      CUDA_CHECK(cudaEventSynchronize(ev_.get()));
    }

    float elapsed_time_from(const ScopedEvent& start) const {
      float ms = {};
      CUDA_CHECK(cudaEventElapsedTime(&ms, start.ev_.get(), ev_.get()));
      return ms;
    }

    cudaEvent_t get() const {
      return ev_.get();
    }

  private:
    detail::UniqueEvent ev_;
  };
} // namespace gpu_lab
