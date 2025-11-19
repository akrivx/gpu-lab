#include <iostream>
#include <random>
#include <stdexcept>
#include <string_view>
#include <cuda_runtime.h>

#include "buffer.hpp"
#include "buffer_view.hpp"
#include "cuda_check.hpp"
#include "with_timer.hpp"
#include "kernels.cuh"

using namespace gpu_lab;


namespace {

  template<typename F>
  auto with_timer_and_stats(
    F&& f,
    cudaStream_t stream = cudaStreamDefault,
    int num_iter = 100,
    int num_warmup_iter = 100
  ) {
    // Warmup
    if (num_warmup_iter > 0) {
      for (int i = 0; i < num_warmup_iter; ++i) {
        f(stream);
      }
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    float min_ms = FLT_MAX;
    float max_ms = 0.0f;
    float total_ms = 0.0f;

    for (int i = 0; i < num_iter; ++i) {
      const auto ms = with_timer(f, stream);
      min_ms = std::min(min_ms, ms);
      max_ms = std::max(max_ms, ms);
      total_ms += ms;
    }

    return std::make_tuple(min_ms, max_ms, total_ms / num_iter);
  }

  template<typename T>
  auto run_memcpy_kernel_and_get_ms(
    DeviceBufferView<const T> src,
    DeviceBufferView<T> dst,
    uint32_t threads_per_block = 1024
  ) {
    if (src.size() % threads_per_block) {
      throw std::runtime_error{"Buffer size must be a multiple of number of threads per block."};
    }
  
    dim3 grid{static_cast<uint32_t>(src.size() / threads_per_block)};
    dim3 block{threads_per_block};

    return with_timer_and_stats([&](cudaStream_t) {
      launch_memcpy_kernel(grid, block, src, dst);
    });
  }

  template<typename T>
  auto run_builtin_memcpy_and_get_ms(
    DeviceBufferView<const T> src,
    DeviceBufferView<T> dst
  ) {
    return with_timer_and_stats([&](cudaStream_t) {
      copy(src, dst);
    });
  }

  void report_stats(
    std::string_view label,
    size_t total_bytes,
    std::tuple<float, float, float> min_max_avg_ms
  ) {
      auto bw_from_ms = [&](float ms) {
        const float sec = ms / 1000.0f;
        return (total_bytes / sec) / 1e9f; // GB/s
      };
  
      const float bw_best  = bw_from_ms(std::get<0>(min_max_avg_ms)); // fastest -> highest BW
      const float bw_worst = bw_from_ms(std::get<1>(min_max_avg_ms)); // slowest -> lowest BW
      const float bw_avg   = bw_from_ms(std::get<2>(min_max_avg_ms));
  
      std::printf(
          "%s\n"
          "  Time (ms):\n"
          "    min  = %.3f\n"
          "    max  = %.3f\n"
          "    avg  = %.3f\n"
          "  Bandwidth (GB/s):\n"
          "    best = %.2f\n"
          "    worst= %.2f\n"
          "    avg  = %.2f\n\n",
          label.data(),
          std::get<0>(min_max_avg_ms),
          std::get<1>(min_max_avg_ms),
          std::get<2>(min_max_avg_ms),
          bw_best,
          bw_worst,
          bw_avg
      );
  }
  

  void run_experiments(DeviceBufferView<const uint4> src, DeviceBufferView<uint4> dst) {
    const auto nbytes = src.size() * sizeof(uint4);

    const auto u8_ms = run_memcpy_kernel_and_get_ms(
      src.as<const uint8_t>(),
      dst.as<uint8_t>()
    );
    report_stats("u8 memcpy", nbytes, u8_ms);
    
    const auto u16_ms = run_memcpy_kernel_and_get_ms(
      src.as<const uint16_t>(),
      dst.as<uint16_t>()
    );
    report_stats("u16 memcpy", nbytes, u16_ms);
    
    const auto u32_ms = run_memcpy_kernel_and_get_ms(
      src.as<const uint32_t>(),
      dst.as<uint32_t>()
    );
    report_stats("u32 memcpy", nbytes, u32_ms);
    
    const auto u32x4_ms = run_memcpy_kernel_and_get_ms(
      src.as_const(),
      dst
    ); 
    report_stats("u32x4 memcpy", nbytes, u32x4_ms);

    const auto u32x4_builtin_ms = run_builtin_memcpy_and_get_ms(
      src.as_const(),
      dst
    );
    report_stats("u32x4 built-in memcpy", nbytes, u32x4_builtin_ms);
  }

}


int main() {

  try {

    constexpr int N = 10 * (2 << 19);
    std::cout << "Copying " << (N * sizeof(uint4) / sizeof(uint8_t)) << " u8s with various methods" << std::endl;

    DeviceBuffer<uint4> d_src{N};
    DeviceBuffer<uint4> d_dst{N};

    {
      std::mt19937 rng(std::random_device{}());
      std::uniform_int_distribution<int> dist{0, 100};
      HostPinnedBuffer<uint32_t> h_buf{N * 4};
      auto h_view = h_buf.view();
      for (size_t i = 0; i < h_view.size(); ++i) {
        h_view[i] = dist(rng);
      }
      copy(h_view, d_src.view_as<uint32_t>());
    }

    run_experiments(d_src.cview(), d_dst.view());
  }
  catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;
  }
  
  return 0;
}
