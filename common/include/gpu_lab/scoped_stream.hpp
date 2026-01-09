#pragma once

#include <memory>

#include <cuda_runtime.h>

#include "gpu_lab/cuda_check.hpp"

namespace gpu_lab {
  namespace detail {
    struct StreamDeleter {
      using pointer = cudaStream_t;
      void operator()(cudaStream_t s) const noexcept {
        CUDA_CHECK_TERMINATE(cudaStreamDestroy(s));
      }
    };

    using UniqueStream = std::unique_ptr<cudaStream_t, detail::StreamDeleter>;

    inline auto make_unique_stream(unsigned int flags) {
      cudaStream_t stream{};
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, flags));
      return UniqueStream{stream};
    }
  } // namespace detail

  class ScopedStream {
  public:
    explicit ScopedStream(unsigned int flags = cudaStreamDefault)
        : stream_{detail::make_unique_stream(flags)} {}

    void sync() const { CUDA_CHECK(cudaStreamSynchronize(stream_.get())); }

    void wait_event(cudaEvent_t event, unsigned int flags = cudaEventWaitDefault) const {
      CUDA_CHECK(cudaStreamWaitEvent(get(), event, flags));
    }

    cudaStream_t get() const { return stream_.get(); }

  private:
    detail::UniqueStream stream_;
  };
} // namespace gpu_lab
