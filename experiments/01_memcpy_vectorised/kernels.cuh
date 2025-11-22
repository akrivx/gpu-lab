#pragma once

#include <cuda_runtime.h>
#include "buffer_view.hpp"

namespace gpu_lab {

  void launch_memcpy_kernel(dim3 grid, dim3 block, DeviceBufferView<const uint8_t> src, DeviceBufferView<uint8_t> dst);
  void launch_memcpy_kernel(dim3 grid, dim3 block, DeviceBufferView<const uint16_t> src, DeviceBufferView<uint16_t> dst);
  void launch_memcpy_kernel(dim3 grid, dim3 block, DeviceBufferView<const uint32_t> src, DeviceBufferView<uint32_t> dst);
  void launch_memcpy_kernel(dim3 grid, dim3 block, DeviceBufferView<const uint4> src, DeviceBufferView<uint4> dst);

}