# Design Document Template

> #### Purpose
> This document serves as a template for design documents in this repository. Design docs describe *semantics* and *intent*, not implementation details or tasks. They are the long-lived reference for why a component exists and how it is meant to behave.

## Overview

### What to include

- A short, high-level description of the component or feature.
- What problem it solves.
- Where it fits in the overall system.

### What to avoid

- Implementation details.
- Historical background or ticket references.

### Example

> `DeviceAsyncArena` provides stream-affine, asynchronous scratch allocation for temporary GPU memory.

## Motivation

### What to include

- Why this component is needed.
- What pain points it addresses.
- What existing approaches are insufficient.

### What to avoid

- Repeating the overview.
- Over-selling or sepculative benefits.

### Example

> Temporary GPU allocations need a clear lifetime model without global synchronisation or event tracking.

## Design Goals

### What to include

- A concise list of **positive goals** (things this design aims to achieve).
- Goals should be concrete and testable.

### What to avoid

- Vague statements ("fast", "flexible").
- Too many goals.

### Example

> - Stream-affine semantics
> - Asynchronous allocation and release
> - Minimal API surface

## Non-Goals

### What to include

- Explicitly state what this design **does not attempt to solve**.
- This is crucial for avoiding scope creep later.

### What to avoid

- Apologetic and defensive explanations on why the design might not offer the best solution that solves all the problems.

### Example

> - Multi-stream lifetime tracking
> - General-purpose memory allocation

## Public API

### What to include

- The intended public interface.
- Function signatures or class definitions.
- This is the contract users rely on.

### What to avoid

- Private helpers.
- Template metaprogramming or implementation tricks

### Example

> ```cpp
> class DeviceAsyncArena {
> public:
>   // ...
>   void* alloc_bytes(std::size_t nbytes, std::size_t alignment);
>   // ...
> };
>
> template<typename T>
> T* alloc(DeviceAsyncArena& arena, std::size_t count);
> ```

## Semantics

### What to include

- Precise behavioural rules.
- Lifetime guarantees.
- Ordering and correctness assumptions.
- How the API behaves in normal use.

### What to avoid

- Code-level algorithms.
- Performance tuning details.

### Example

> Memory allocated from the arena is valid until `reset()` or destruction and must only be used on the arena's stream.

## Constraints and Invariants

### What to include

- Rules that _must always hold_.
- Invariants enforced by convetion or debug checks.
- Things users must not do.

### What to avoid

- Repeating goals.
- Weak language ("should", "ideally").

### Example

> - All allocations and frees are ordered on a single CUDA stream.
> - Nested launch wrapper calls are prohibited.

## Rationale

### What to include

- Why key design decisions were made.
- Trade-offs that were considered.
- Why simpler or more general designs were rejected.

### What to avoid

- Re-arguing settled decisions.
- Implementation-leve justifications.

### Example

> Stream-affinity avoids the need for event-based lifetime tracking and keeps semantics simple.

## Usage Examples

### What to include

- Minimal, idiomatic examples.
- Demonstrate intended usage patterns.

### What to avoid

- Exhaustive examples
- Edge-case handling.

### Example

> ```cpp
> launch_kernel("axpby_kernel", axpby_kernel, grid, block, 0, stream, a, x, b, y, out, n);
> ```

## Interaction With Other Components

### What to include

- How this component fits with existing systems.
- Dependencies or assumptions about other modules.

### What to avoid

- Deep architectural diagrams unless necessary.

### Example

> This component complements `DeviceBuffer` by handling temporary allocations only.

## Future Work (Out of Scope)

### What to include

- Plausible extensions or optimisations.
- Clearly marked as **not part of the current design**.

### What to avoid

- Promises or commitments.
- Detailed designs for future features.

### Example

> - Chunked allocation
> - Allocation reuse
> - Advanced profiling hooks

## Final Notes [Optional]

### What to include

- Any clarifying remarks.
- Design philosophy or guiding principles.

### What to avoid

- Repetition.
- Apologies.

---

## Design Doc Guidelines

- Design docs describe **what and why, not how**.
- Keep them concise; update them only when semantics change.
- If implementation diverges from the doc, the doc is wrong.
- Issues and PRs should reference design docs, not duplicate them.
- Design docs should be versioned with the code and reviewed like any other change.
