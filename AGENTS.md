# JustPIC.jl contributor guidance

## Project

JustPIC.jl is a Julia Particle-in-Cell library for advection and particle/grid
coupling in 2D and 3D geodynamics applications. The same KernelAbstractions
implementation runs on the CPU and, through package extensions, on CUDA,
AMDGPU, and Metal backends.

The package is Julia 1.10+ and its primary dependencies include
KernelAbstractions, CellArrays, CellArraysIndexing, StaticArrays,
ImplicitGlobalGrid, MPI, GridGeometryUtils, and JLD2.

## Repository map

- `src/JustPIC.jl` defines the module, core types, and backend allocation helpers.
- `src/common.jl` includes the shared implementation and exports the public API.
- `src/Particles/` contains particle storage, injection, movement, advection,
  and semi-Lagrangian schemes.
- `src/Interpolations/` contains particle-to-grid, grid-to-particle, and
  centroid interpolation.
- `src/MarkerChain/` contains free-surface/topography marker chains.
- `src/PassiveMarkers/` contains passive tracer support.
- `src/PhaseRatios/` contains phase-ratio bookkeeping.
- `src/CellArrays/` contains cell-array integration and conversions.
- `src/IO/` contains JLD2 checkpointing.
- `ext/` contains only backend-specific allocation and conversion extensions.
- `test/` contains the CPU/GPU test files and `test/runtests.jl`, the explicit
  test runner.
- `docs/src/` contains the Documenter manual; `docs/examples/` contains example
  scripts.
- `.agents/` contains focused testing, documentation, and validation notes.

## Implementation rules

- Keep shared algorithms in `src/`; do not duplicate them in `ext/`.
- Write backend-generic kernels with KernelAbstractions `@kernel` and `@index`.
  Use `CAI.@index` for CellArraysIndexing access.
- Launch kernels through `launch!` and obtain the backend with `ka_backend(x)`.
  Do not branch on `Array`, `CuArray`, or vendor-specific array types in shared
  code.
- Kernels must be allocation-free and type-stable. Prefer tuples and
  `SVector`s over temporary arrays, and keep helpers `@inline` where useful.
- Keep numeric code generic in the element type. In particular, Metal does not
  support `Float64`; avoid hard-coded `Float64` literals in kernels.
- Treat particle occupancy as an invariant across injection, cleanup, and
  `move_particles!`. Read `src/Particles/move_safe.jl` before changing any of
  those paths.
- When adding an exported function, update the export list in `src/common.jl`
  and its API documentation in `docs/src/API.md`.
- Do not change dependencies or compatibility bounds unless the task requires
  it. Use Runic for formatting, not JuliaFormatter.

## Tests and documentation

Run the CPU suite with:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

The runner accepts a backend argument for GPU testing, for example:

```sh
julia --project=. -e 'using Pkg; Pkg.test(; test_args=["--backend=CUDA"])'
```

When adding a test file, register it in `test/runtests.jl`; files are not
discovered automatically. Prefer CPU-first debugging for GPU failures and
avoid scalar indexing of device arrays in new tests.

Build the manual with:

```sh
julia --project=docs docs/make.jl
```

Read `.agents/testing.md`, `.agents/documentation.md`, and
`.agents/validation.md` when the change touches those areas.

## Workflow

Keep changes focused, preserve unrelated work in the working tree, update tests
and documentation with behavior changes, and follow the project’s
[contribution guidelines](CONTRIBUTING.md). Before opening a PR, format the
changed Julia code with Runic and run the most relevant CPU tests.
