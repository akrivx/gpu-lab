#include <cstdio>
#include <cuda_runtime.h>

int main() {
  int dev = 0;  // single-GPU assumption, device 0

  if (cudaSetDevice(dev) != cudaSuccess) {
    std::printf("512\n");
    return 0;
  }

  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
    std::printf("512\n");
    return 0;
  }

  std::printf("%zu\n", static_cast<std::size_t>(prop.texturePitchAlignment));
  return 0;
}
