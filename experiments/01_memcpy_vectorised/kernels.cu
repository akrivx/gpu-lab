#include "kernels.cuh"


namespace {

  template<typename T>
  __global__ void memcpy_kernel(const T* __restrict__ src, T* __restrict__ dst, size_t count) {
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
      dst[i] = src[i];
    }
  }

}


namespace gpu_lab {

  void launch_memcpy_kernel(dim3 grid, dim3 block, DeviceBufferView<const uint8_t> src, DeviceBufferView<uint8_t> dst) {
    memcpy_kernel<<<grid, block>>>(src.data(), dst.data(), dst.size());
  }

  void launch_memcpy_kernel(dim3 grid, dim3 block, DeviceBufferView<const uint16_t> src, DeviceBufferView<uint16_t> dst) {
    memcpy_kernel<<<grid, block>>>(src.data(), dst.data(), dst.size());
  }

  void launch_memcpy_kernel(dim3 grid, dim3 block, DeviceBufferView<const uint32_t> src, DeviceBufferView<uint32_t> dst) {
    memcpy_kernel<<<grid, block>>>(src.data(), dst.data(), dst.size());
  }

  void launch_memcpy_kernel(dim3 grid, dim3 block, DeviceBufferView<const uint4> src, DeviceBufferView<uint4> dst) {
    memcpy_kernel<<<grid, block>>>(src.data(), dst.data(), dst.size());
  }

}
