#pragma once

#include <cstdio>
#include <exception>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

#include <cuda_runtime.h>

namespace gpu_lab {
  namespace detail {
    struct CudaErrorInfo {
      cudaError_t err;
      const char* expr;
      const char* file;
      int line;

      void write(FILE* fp) const noexcept {
        std::fprintf(fp,
                     "FATAL CUDA ERROR: %s: (%s)\n  at %s:%d\n  expr: %s\n",
                     cudaGetErrorName(err),
                     cudaGetErrorString(err),
                     file,
                     line,
                     expr);
      }

      std::string to_string() const {
        return std::format("{} ({})\n  at {}:{}\n  expr: {}\n",
                           cudaGetErrorName(err),
                           cudaGetErrorString(err),
                           file,
                           line,
                           expr);
      }
    };

    [[noreturn]] inline void cuda_terminate(CudaErrorInfo e) noexcept {
      e.write(stderr);
      std::fflush(stderr);
      std::terminate();
    }
  } // namespace detail

  class CudaError : public std::runtime_error {
  public:
    CudaError(detail::CudaErrorInfo e)
        : std::runtime_error{e.to_string()} {}
  };

  namespace detail {
    inline void
    cuda_check_terminate(cudaError_t err, const char* expr, const char* file, int line) noexcept {
      if (err != cudaSuccess) {
        cuda_terminate({err, expr, file, line});
      }
    }

    inline void cuda_check_throw(cudaError_t err, const char* expr, const char* file, int line) {
      if (err != cudaSuccess) {
        throw CudaError{CudaErrorInfo{err, expr, file, line}};
      }
    }
  } // namespace detail
} // namespace gpu_lab

#define CUDA_CHECK(expr) gpu_lab::detail::cuda_check_throw((expr), #expr, __FILE__, __LINE__)

#define CUDA_CHECK_TERMINATE(expr)                                                                 \
  gpu_lab::detail::cuda_check_terminate((expr), #expr, __FILE__, __LINE__)
