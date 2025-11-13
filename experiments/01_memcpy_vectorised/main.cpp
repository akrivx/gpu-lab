#include <cuda_runtime.h>
#include <iostream>

int main() {
  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
    std::cout << "Running on GPU: " << prop.name << "\n";
  } else {
    std::cerr << "Failed to query device properties\n";
  }
  return 0;
}
