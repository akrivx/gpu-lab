#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <cuda_runtime.h>


namespace gpu_lab {

  class CudaError : public std::runtime_error {
  public:
    CudaError(cudaError_t err, std::string_view msg)
      : std::runtime_error{
          std::string(cudaGetErrorName(err)) + ": " +
          cudaGetErrorString(err) + " | " + std::string{msg}} {}
  };

  namespace detail {
    inline void cuda_check_impl(cudaError_t err, std::string_view msg) {
      if (err != cudaSuccess) {
        throw CudaError{err, msg};
      }
    }
  }

}

#define CUDA_CHECK(expr) \
  gpu_lab::detail::cuda_check_impl((expr), std::string_view{__FILE__":" + std::to_string(__LINE__)})