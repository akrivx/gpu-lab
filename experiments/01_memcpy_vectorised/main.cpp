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
#include "experiment.hpp"
#include "kernels.cuh"

using namespace gpu_lab;

namespace {
  template<typename T>
  auto run_memcpy_kernel_and_get_ms(
    DeviceBufferView<const T> src,
    DeviceBufferView<T> dst,
    uint32_t threads_per_block = 1024)
  {
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
    DeviceBufferView<T> dst)
  {
    return run_benchmark([&](cudaStream_t) {
      copy(src, dst);
    });
  }
} // namespace (anonymous)

int main() {
  try {
    constexpr int N = 10 * (2 << 19);

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

    auto src = d_src.cview();
    auto dst = d_dst.view();

    run_bandwidth_experiments({
        {"u8",      [&]() { return run_memcpy_kernel_and_get_ms(src.as<const uint8_t>(), dst.as<uint8_t>()); }},
        {"u16",     [&]() { return run_memcpy_kernel_and_get_ms(src.as<const uint16_t>(), dst.as<uint16_t>()); }},
        {"u32",     [&]() { return run_memcpy_kernel_and_get_ms(src.as<const uint32_t>(), dst.as<uint32_t>()); }},
        {"uint4",   [&]() { return run_memcpy_kernel_and_get_ms(src.as_const(), dst); }},
        {"builtin", [&]() { return run_builtin_memcpy_and_get_ms(src.as_const(), dst); }},
      },
      N * sizeof(uint4),
      std::cout);
  }
  catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;
  }
  return 0;
}
