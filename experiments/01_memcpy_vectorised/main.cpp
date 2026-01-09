#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "gpu_lab/buffer.hpp"
#include "gpu_lab/buffer_view.hpp"
#include "gpu_lab/cuda_check.hpp"
#include "gpu_lab/experiment.hpp"
#include "gpu_lab/with_timer.hpp"

#include "kernels.cuh"

using namespace gpu_lab;

namespace {
  template <typename T>
  auto run_memcpy_kernel_and_get_ms(DeviceBufferView<const T> src,
                                    DeviceBufferView<T> dst,
                                    uint32_t threads_per_block = 1024) {
    if (src.size() % threads_per_block) {
      throw std::runtime_error{"Buffer size must be a multiple of number of threads per block."};
    }

    dim3 grid{static_cast<uint32_t>(src.size() / threads_per_block)};
    dim3 block{threads_per_block};

    return run_benchmark([&](cudaStream_t) { launch_memcpy_kernel(grid, block, src, dst); });
  }

  template <typename T>
  auto run_builtin_memcpy_and_get_ms(DeviceBufferView<const T> src, DeviceBufferView<T> dst) {
    return run_benchmark([&](cudaStream_t) { copy(src, dst); });
  }
} // namespace

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
      copy(h_view, view_as<uint32_t>(d_src));
    }

    auto src = d_src.cview();
    auto dst = d_dst.view();

    run_bandwidth_experiments(
        {
            {"u8",
             [&]() {
               return run_memcpy_kernel_and_get_ms(view_as<const uint8_t>(d_src),
                                                   view_as<uint8_t>(d_dst));
             }},
            {"u16",
             [&]() {
               return run_memcpy_kernel_and_get_ms(view_as<const uint16_t>(d_src),
                                                   view_as<uint16_t>(d_dst));
             }},
            {"u32",
             [&]() {
               return run_memcpy_kernel_and_get_ms(view_as<const uint32_t>(d_src),
                                                   view_as<uint32_t>(d_dst));
             }},
            {"uint4", [&]() { return run_memcpy_kernel_and_get_ms(d_src.cview(), d_dst.view()); }},
            {"builtin",
             [&]() { return run_builtin_memcpy_and_get_ms(d_src.cview(), d_dst.view()); }},
        },
        N * sizeof(uint4),
        std::cout);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;
  }
  return 0;
}
