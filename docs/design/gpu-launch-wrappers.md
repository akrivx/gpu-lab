# GPU Launch Wrappers

## Overview

This document defines a lightweight, compile-time configurable mechanism for wrapping GPU work submission.

The wrappers provide optional instrumentation and policy enforcement at a single choke point, without requiring runtime configuration or passing launcher objects.

## Design Goals

- Free functions (no launcher objects)
- Compile-time configuration
- Zero-cost when disabled
- Uniform handling of:
  - direct kernel launches
  - indirect/composite GPU operations (libraries, helper functions)
- Simple call sites
- Non-reentrant semantics (no nesting)

Non-goals:
- Enforcing exclusive usage
- Capturing internal kernels launched by libraries
- Runtime feature toggles

## Public API

```cpp
// Direct kernel launch (Kernel must be a __global__ function)
template <typename Kernel, typename... Args>
void launch_kernel(const char* name,
                   Kernel&& kernel,
                   dim3 grid, dim3 block, std::size_t shmem, cudaStream_t stream,
                   Args&&... args);

// Indirect/composite GPU operation
template <typename F, typename... Args>
decltype(auto) launch_op(const char* name, F&& f, cudaStream_t stream, Args&&... args);
```

## Semantics

`launch_kernel`:

- Wraps a CUDA kernel launch (`<<<...>>>`)
- Launches on the provided stream
- Optional instrumentation is applied according to the configured policy (compile-time based on macros)
- Does not synchronie unless explicitly enabled by policy

`launch_op`:

- Wraps a callable that enqueues GPU work on the provided stream
- Supports callables of the form `f(cudaStream_t, Args...)` or `f(Args...)`
  - The implementation detects (at compile-time) the proper way of invoking the callable.
- Intended for library calls or composite operations that submit GPU work (multiple kernels, copies, memsets, etc.)
- Callable is responsible for binding library handles to the stream
- Instrumentation spans the entire callable
- Does not synchronise unless explicitly enabled the policy

## Reentrancy / Nesting

### Rule

`launch_kernel` and `launch_op` are non-reentrant.

- Calling either wrapper from within an active wrapper region is **invalid**.
- Nesting is prohibited regardless of wrapper type:
  - `launch_kernel` → `launch_kernel`
  - `launch_kernel` → `launch_op`
  - `launch_op` → `launch_kernel`
  - `launch_op` → `launch_op`
 
### Detection

Implemented via a lightweight **thread-local guard**.

### Behaviour

- **Debug builds**: assert or throw with a clear diagnostic
- **Release builds**: check may be compiled out

### Rationale

Keeps wrapper semantics and implementation simple and unambiguous

## Usage Examples

### Direct kernel launch

```cpp
launch_kernel("axpby_kernel", axpby_kernel, grid, block, 0, stream, a, x, b, y, out, n);

// Or:

LAUNCH_KERNEL(axpby_kernel, grid, block, 0, stream, a, x, b, y, out, b);
```

### Indirect GPU operations

```cpp
launch_op("cublasGemmEx", stream, [&] {
  cublasSetStream(handle, stream);
  cublasGemmEx(handle, ...);
});

// Or:

auto gemm = [](auto stream, auto handle) {
  cublasSetStream(handle, stream);
  return cublasGemmEx(handle, ...);
};

auto res = LAUNCH_OP(gemm, stream, handle);
```

```cpp
int decode_image(cudaStream_t stream, ImageView img, ImagView out);

// launch_op detects if the wrapped op must be passed a stream before its args
auto res = launch_op("decode_image", decode_image, stream, img, out);

// Or:

auto res = LAUNCH_OP(decode_image, stream, img, out);
```

## Future Work (out of scope)

- CUDA Graph capture integration
- Advanced profiling or metadata collection

## Final Note

These wrappers define a **clear and explicit GPU submission model**.

They intentionally trade generality for:

- clarity,
- predictability,
- and ease of reasoning.

This constraint is deliberate.







