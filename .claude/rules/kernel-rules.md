---
paths:
  - src/**/*.jl
  - ext/**/*.jl
---

# Kernel Function Rules

JustPIC kernels are written once with KernelAbstractions and run on CPU, CUDA, AMDGPU and
Metal. Everything below exists to keep that true.

## Requirements

- Use `@kernel function foo_kernel!(...)` + KA's `@index`; launch with
  `launch!(ka_backend(x), foo_kernel!, ndrange, args...)` — `launch!` synchronizes, which
  callers (host reads, MPI halos, injection) rely on
- Cell-element access uses `CAI.@index p[ip, I...]` from CellArraysIndexing. KA's `@index`
  and CellArraysIndexing's `@index` only coexist because the latter is imported qualified
  (`import CellArraysIndexing as CAI`) — never `using CellArraysIndexing` unqualified
- Kernels must be **type-stable** and **allocation-free**: `SVector`s and tuples, no `push!`,
  no dynamic dispatch, no closures capturing non-concrete values
- Mark functions called inside kernels `@inline`; prefer `ifelse` over branchy `if`/`else`
  on data values
- No `error()`/string interpolation inside kernels
- **Never loop over grid cells outside a kernel** — that code silently becomes
  CPU-only

## Backend genericity

- No backend-specific code in `src/`: no `CuArray`, no `Array(x)` conversions, no
  `if backend == ...`. Recover the backend with `ka_backend(x)` and dispatch
- `ext/` extensions supply *only* allocation (`TA`, `CA`, `undef_cell_array`) and
  host↔device conversion methods. Algorithm changes never go in `ext/`
- Allocate particle-style storage with `cell_array(backend, x, ncells, ni)`

## Numeric types (Metal has no Float64)

- No `0.0`/`1.0` literals in kernels — use `zero(T)`, `one(T)`, `convert(T, x)` or rational
  literals; take `T` from the arrays being processed
- Integrator constants go through `set_precision(integrator, T)`
- A coordinate *range* passed directly into a kernel must go through `recast_grid`
  (Base ranges index through Float64 internally; see [src/launch.jl](../../src/launch.jl))

## Particle storage invariants

- `Particles.index` is the per-cell occupancy mask; any kernel that writes particle slots
  must keep mask and coordinate slots consistent
- Occupancy logic (injection, cleanup, `move_particles!`) is coupled — read
  [src/Particles/move_safe.jl](../../src/Particles/move_safe.jl) before modifying any of it
