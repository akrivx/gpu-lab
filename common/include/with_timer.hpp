#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <tuple>

#include "scoped_event.hpp"


namespace gpu_lab {

  /// @brief Measures the GPU execution time of a callable using CUDA events.
  /// 
  /// This helper records a start and stop CUDA event around the invocation of
  /// the provided callable. The callable must accept a single parameter of type
  /// `cudaStream_t` and enqueue all work onto that stream. The function waits
  /// for the stop event to complete and then returns the elapsed time in
  /// milliseconds.
  ///
  /// Typical usage:
  /// @code
  /// float ms = with_timer([&](cudaStream_t s) {
  ///   my_kernel<<<grid, block, 0, s>>>(...);
  /// });
  /// @endcode
  ///
  /// @tparam F A callable with signature `void f(cudaStream_t)`
  /// @param f The function whose GPU execution time is to be measured
  /// @param stream The CUDA stream on which the callable will be invoked
  /// @return Elapsed GPU time in milliseconds
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

  /// @brief Measures min, max, and average GPU execution time over multiple runs.
  ///
  /// This helper function runs a callable several times, measuring the elapsed
  /// GPU time (in milliseconds) of each run using `with_timer()`. Optional warm-up
  /// iterations can be executed beforehand to amortize one-off effects such as
  /// first-time kernel compilation or cache population.
  ///
  /// The callable must accept a single parameter of type `cudaStream_t` and
  /// enqueue all its work onto that stream. After warm-up iterations,
  /// the given stream is synchronised to ensure all warm-up activity has completed
  /// before measurement begins.
  ///
  /// @tparam F A callable with signature `void(cudaStream_t)`
  /// @param f The function to benchmark
  /// @param stream The CUDA stream used for all invocations
  /// @param num_iter Number of timed iterations to run (must be > 0)
  /// @param num_warmup_iter Number of warm-up iterations to run before timing (use <= 0 to suppress)
  /// @return A tuple `(min_ms, max_ms, avg_ms)` containing:
  ///   - `min_ms`: minimum elapsed GPU time over all timed runs
  ///   - `max_ms`: maximum elapsed GPU time over all timed runs
  ///   - `avg_ms`: average elapsed GPU time over all timed runs
  template<typename F>
  auto run_benchmark(
    F f,
    cudaStream_t stream = cudaStreamDefault,
    int num_iter = 50,
    int num_warmup_iter = 10
  ) {
    // Warmup
    if (num_warmup_iter > 0) {
      for (int i = 0; i < num_warmup_iter; ++i) {
        f(stream);
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    float min_ms   = std::numeric_limits<float>::max();
    float max_ms   = 0.0f;
    float total_ms = 0.0f;

    for (int i = 0; i < num_iter; ++i) {
      const auto ms = with_timer(f, stream);
      min_ms = std::min(min_ms, ms);
      max_ms = std::max(max_ms, ms);
      total_ms += ms;
    }

    return std::make_tuple(min_ms, max_ms, total_ms / num_iter);
  }

}
