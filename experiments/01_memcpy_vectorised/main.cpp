#include <iostream>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <functional>

#include <cuda_runtime.h>

#include "buffer.hpp"
#include "buffer_view.hpp"
#include "cuda_check.hpp"
#include "with_timer.hpp"
#include "kernels.cuh"

using namespace gpu_lab;

namespace {

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

    return run_benchmark([&](cudaStream_t) {
      launch_memcpy_kernel(grid, block, src, dst);
    });
  }

  template<typename T>
  auto run_builtin_memcpy_and_get_ms(
    DeviceBufferView<const T> src,
    DeviceBufferView<T> dst
  ) {
    return run_benchmark([&](cudaStream_t) {
      copy(src, dst);
    });
  }

  void run_experiments(DeviceBufferView<const uint4> src, DeviceBufferView<uint4> dst) {
    const auto total_bytes = src.size() * sizeof(uint4);

    const auto gbps_from_ms = [=](float ms) {
      const float sec = ms / 1000.0f;
      return (total_bytes / sec) / 1e9f; // GB/s
    };

    const std::vector<std::pair<std::string, std::function<std::tuple<float, float, float>()>>> experiments = {
      {"u8",      [&]() { return run_memcpy_kernel_and_get_ms(src.as<const uint8_t>(), dst.as<uint8_t>()); }},
      {"u16",     [&]() { return run_memcpy_kernel_and_get_ms(src.as<const uint16_t>(), dst.as<uint16_t>()); }},
      {"u32",     [&]() { return run_memcpy_kernel_and_get_ms(src.as<const uint32_t>(), dst.as<uint32_t>()); }},
      {"uint4",   [&]() { return run_memcpy_kernel_and_get_ms(src.as_const(), dst); }},
      {"builtin", [&]() { return run_builtin_memcpy_and_get_ms(src.as_const(), dst); }},
    };

    const char* sep = 
      "+---------+------------+------------+------------+------------+------------+------------+\n";

    std::printf("%s", sep);
    std::printf(
      "|  Type   |  Min ms    |  Avg ms    |  Max ms    | Best GB/s  | Avg GB/s   | Worst GB/s |\n"
    );
    std::printf("%s", sep);

    for (const auto& [type_label, f]: experiments) {
      const auto [min_ms, max_ms, avg_ms] = f();

      const float best_gbps  = gbps_from_ms(min_ms); // fastest -> highest BW
      const float worst_gbps = gbps_from_ms(max_ms); // slowest -> lowest BW
      const float avg_gbps   = gbps_from_ms(avg_ms);
      std::printf(
        "| %-7s | %10.3f | %10.3f | %10.3f | %10.1f | %10.1f | %10.1f |\n",
        type_label.c_str(),
        min_ms,
        avg_ms,
        max_ms,
        best_gbps,
        avg_gbps,
        worst_gbps
      );
    }

    std::printf("%s", sep);
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
