#pragma once

#include <optional>
#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "scoped_event.hpp"


namespace gpu_lab {

  template<typename F>
  auto with_scoped_timer(F&& f, cudaStream_t stream = cudaStreamDefault) {
    ScopedEvent start;
    ScopedEvent stop;
    start.record();
    f();
    stop.record();
    stop.sync();
    float ms = {};
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.e_, end.e_));
    return ms;
  }

}