#pragma once

#include <cuda_runtime.h>
#include "scoped_event.hpp"


namespace gpu_lab {

  template<typename F>
  auto with_timer(F&& f, cudaStream_t stream = cudaStreamDefault) {
    ScopedEvent start;
    ScopedEvent stop;
    start.record(stream);
    f(stream);
    stop.record(stream);
    stop.sync();
    return stop.elapsed_time_from(start);
  }

}
