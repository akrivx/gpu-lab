#include <iostream>
#include <random>
#include <stdexcept>
#include <string_view>
#include <cuda_runtime.h>

#include "buffer.hpp"
#include "buffer_view.hpp"
#include "with_timer.hpp"
#include "kernels.cuh"

using namespace gpu_lab;


namespace {

  template<typename T>
  auto run_kernel_and_get_ms(
    DeviceBufferView<const T> src,
    DeviceBufferView<T> dst,
    uint32_t threads_per_block = 1024,
    int num_iter = 100
  ) {
    if (src.size() % threads_per_block) {
      throw std::runtime_error{"Buffer size must be a multiple of number of threads per block."};
    }

    constexpr int NUM_WARM_UP_ITER = 100;

    dim3 grid{static_cast<uint32_t>(src.size() / threads_per_block)};
    dim3 block{threads_per_block};

    // Warmup
    for (int i = 0; i < NUM_WARM_UP_ITER; ++i) {
      launch_memcpy_kernel(grid, block, src, dst);
    }

    float min_ms = FLT_MAX;
    float max_ms = 0.0f;
    float avg_ms = 0.0f;

    for (int i = 0; i < num_iter; ++i) {
      const auto ms = with_timer([&](cudaStream_t) {
        launch_memcpy_kernel(grid, block, src, dst);
      });
      min_ms = std::min(min_ms, ms);
      max_ms = std::max(max_ms, ms);
      avg_ms += ms;
    }

    avg_ms /= num_iter;
    return std::make_tuple(min_ms, max_ms, avg_ms);
  }

  void print_timings(std::string_view header, float min_ms, float max_ms, float avg_ms) {
    std::cout << header << "(min=" << min_ms << ", max=" << max_ms << ", avg=" << avg_ms << ") [ms]" << std::endl;
  }

  void print_timings(std::string_view header, std::tuple<float, float, float> min_max_avg_ms) {
    print_timings(
      header,
      std::get<0>(min_max_avg_ms),
      std::get<1>(min_max_avg_ms),
      std::get<2>(min_max_avg_ms)
    );
  }

  void run_experiments(DeviceBufferView<const uint4> src, DeviceBufferView<uint4> dst) {
    const auto u8_ms = run_kernel_and_get_ms(
      src.as<const uint8_t>(),
      dst.as<uint8_t>()
    );
    print_timings("u8 copy: ", u8_ms);
    
    const auto u16_ms = run_kernel_and_get_ms(
      src.as<const uint16_t>(),
      dst.as<uint16_t>()
    );
    print_timings("u16 copy: ", u16_ms);
    
    const auto u32_ms = run_kernel_and_get_ms(
      src.as<const uint32_t>(),
      dst.as<uint32_t>()
    );
    print_timings("u32 copy: ", u32_ms);
    
    const auto u32x4_ms = run_kernel_and_get_ms(
      src.as_const(),
      dst
    ); 
    print_timings("u32x4 copy: ", u32x4_ms);
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
