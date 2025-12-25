# Device Async Arena

## Overview

`DeviceAsyncArena` is a stream-affine scratch allocator for temporary GPU memory.

It provides async allocation and reclamation of device memory using CUDA's stream-ordered allocation APIs. The arena is intended for **short-lived scratch buffers** used during kernel execution, not for persistent storage.

## Design Constraints

- **Stream-affine**
  - All allocations and frees are ordered on a single CUDA stream.
  - Returned pointers must only be used on that stream (or an equivalent ordering).
- **Async semantics**
  - No device-wide synchronisation.
  - Uses `cudaMallocAsync` / `cudaFreeAsync`.
- **Explicit lifetime**
  - Scratch lifetime ends at `reset()` or arena destruction.
  - No attempt to infer kernel usage.
- **Mechanics hidden**
  - Backing strategy is an implementation detail.
  - Allows future optimisation (e.g. chunking) without API changes.

## Public API

```cpp
class DeviceAsyncArena {
public:
  explicit DeviceAsyncArena(cudaStream_t stream, std::optional<cudaMemPool_t> pool = {});
  void* alloc_bytes(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t));
  void reset();
  cudaStream_t stream() const;
};

template<typename T>
DeviceBufferView<T> alloc(DeviceAsyncArena& arena, std::size_t count);
```

## Semantics

### Allocation

- `alloc()` enqueues `cudaMallocAsync` on the arena's stream.
- Allocation is non-blocking and stream-ordered.

### Usage

- Returned pointers are valid for kernels launched on the same stream.
- Cross-stream usage is explicitly unsupported.

### Reset

- `reset()` enqueues `cudaFreeAsync` for all allocations so far.
- Frees are ordered after all prior work in the stream.
- Reset defines a clear phase boundary for scratch memory.

### Destruction

- Destructor behaves like `reset()`.
- Frees are enqueued asynchronously.
- No blocking or synchronisation.

## Correctness Rationale

Because allocation, kernel execution, and free all occur on the **same stream**, CUDA stream ordering guarantees:

- memory is not freed until all prior kernels using it have completed;
- no explicit events or deferred free queues are required.

This design deliberately avoids multi-stream complexity.

## Current Implementation

- Backing storage: **free-list of async allocations**.
- Each allocation tracked internally and freed on `reset()` / destruction.
- No reuse or suballocation in this version.

## Design Decisions

### Why stream-affine?

It eliminates the need for:

- kernel provenance tracking
- last-use stream inference
- cross-stream fencing
- deferred free queues

### Why not chunking initially?

- Free-list implementation is simpler and correct.
- Optimisation can be added later without changing semantics.
- Arena API is designed to hide allocation strategy.

## Future Evolution (out of scope)

- Chunked / bump allocation
- Allocation reuse
- Size-class caching
- Per-stream arena pools
- Debug instrumentation and stats

## Usage Example

```cpp
DeviceAsyncArena scratch{stream};

auto tmpA = alloc<float>(scratch, n);
auto tmpB = alloc<int>(scratch, m);

kernel<<<grid, block, 0, stream>>>(tmpA, tmpB);

scratch.reset(); // async, stream-ordered
```

RAII usage is also valid:

```cpp
{
  DeviceAsyncArena scratch{stream};
  auto tmp = alloc<float>(scratch, n);
  kernel<<<..., stream>>>(tmp);
} // frees enqueued asynchronously
```

## Final Note

This arena is intentionally **not** a general allocator. It is a small, explicit, and predictable tool for scratch memory.

That constraint is a feature, not a limitation.







