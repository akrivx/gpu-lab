#pragma once

#include <cuda_runtime.h>
#include "noncopyable.hpp"
#include "cuda_check.hpp"


namespace gpu_lab {
  
  struct ScopedEvent : NonCopyable {
    explicit ScopedEvent(unsigned int flags = cudaEventDefault) {
      CUDA_CHECK(cudaEventCreateWithFlags(&e_, flags));
    }

    ~ScopedEvent() {
      CUDA_CHECK(cudaEventDestroy(e_));
    }

    void record(cudaStream_t stream = cudaStreamDefault) const {
      CUDA_CHECK(cudaEventRecord(e_, stream));
    }

    void sync() const {
      CUDA_CHECK(cudaEventSynchronize(e_));
    }

    cudaEvent_t e_;
  };

}
