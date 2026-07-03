# JustPIC.jl — Agent Rules

## Project Overview

JustPIC.jl is a Julia Particle-in-Cell (PIC) advection library for geodynamic simulations,
running on CPU (multithreaded), Nvidia (CUDA), AMD (AMDGPU) and Apple (Metal) GPUs from the
same source. It provides particle containers, advection integrators (Euler/RK2/RK4),
particle↔grid interpolation, phase-ratio bookkeeping, marker chains (free-surface tracking)
and passive tracers.

## Language & Environment

- **Julia 1.10+** | CPU and GPU (CUDA, AMDGPU, Metal)
- **Key packages**: KernelAbstractions.jl, CellArrays.jl, CellArraysIndexing.jl,
  StaticArrays.jl, ImplicitGlobalGrid.jl (MPI halos), JLD2.jl (checkpointing)
- **Formatter**: Runic (`git runic main`), *not* JuliaFormatter

## Architecture (read before editing `src/`)

The package is a single flat module built around backend-generic KernelAbstractions kernels:

- [src/JustPIC.jl](src/JustPIC.jl) is the top module. It defines the shared structs
  (`Particles`, `MarkerChain`, `PassiveMarkers` in `src/particles.jl`, `PhaseRatios` in
  `src/PhaseRatios/`), the advection integrator types (`src/Advection/types.jl`), and then
  includes [src/common.jl](src/common.jl) **once**.
- [src/common.jl](src/common.jl) is the algorithm catalog: it chains `include`s for every
  feature area and re-exports the public API. Kernels are dimension-generic (2D and 3D from
  the same code), so there are no `_2D`/`_3D` submodules — users just call `using JustPIC`.
- **Backend dispatch is by KernelAbstractions backend type**, picked from the array at
  runtime: `launch!(ka_backend(x), kernel!, ndrange, args...)` (see
  [src/launch.jl](src/launch.jl)). JustPIC defines no backend tags of its own; it dispatches
  on `CPU`, `CUDA.CUDABackend`, `AMDGPU.ROCBackend`, `Metal.MetalBackend`.
- [ext/](ext/) extensions (loaded via `[weakdeps]`/`[extensions]`) supply **only**
  backend-specific allocation (`TA`, `CA`, `undef_cell_array`) and host↔device conversion
  (e.g. `CUDA.CuArray(::Particles)`). They do **not** re-include `common.jl` and do not
  forward functions. Never add generic algorithm code to `ext/`.
- Particles are stored in `CellArray`s (`src/CellArrays/`): each grid cell owns a
  fixed-capacity slot of particles (`nxcell`/`min_xcell`/`max_xcell`); `Particles.index` is a
  per-cell boolean occupancy mask. `move_particles!`
  ([src/Particles/move_safe.jl](src/Particles/move_safe.jl)) restores cell locality after
  advection — occupancy invariants span injection, cleanup and move together, so read that
  file before touching any of them.

## Critical Rules

### Kernels (GPU compatibility)

- Use `@kernel` / `@index` (KernelAbstractions). Element access inside cells uses
  `CAI.@index` (CellArraysIndexing) — the two `@index` macros coexist only because
  `CellArraysIndexing` is imported qualified. Keep it that way.
- Kernels must be **type-stable** and **allocation-free**; use `SVector`/tuples, not arrays
- Use `ifelse` or short-circuit-free branching on data; mark helpers `@inline`
- Launch through `launch!` — never `for` loops over grid cells outside kernels, never
  backend-specific launch code in `src/`
- `launch!` is synchronous by design (host reads, MPI halos and injection assume it);
  don't remove the `synchronize` without auditing every caller
- **Metal has no Float64**: keep kernels generic in the eltype (`zero(T)`, `convert(T, x)`,
  rational literals), use `set_precision` for integrator constants and `recast_grid` when a
  coordinate range is passed directly into a kernel

### Types & Memory

- All structs concretely typed; type annotations are for **dispatch**, not documentation
- Use `ka_backend(x)` to recover the backend — never `Array(...)`/`CuArray(...)` branches
  in `src/`
- Allocation helpers: `cell_array(backend, x, ncells, ni)` for filled cell arrays,
  `CA(backend, dims)` for raw ones, `TA(backend)` for the plain array type

### Dependencies

- Never add, remove, or change `[deps]`/`[weakdeps]` in `Project.toml` unless the task
  absolutely requires it; only touch `[compat]` when explicitly asked

## Naming Conventions

- **Files**: snake_case, mirrored per feature area (`Particles/`, `MarkerChain/`, …)
- **Types**: PascalCase (`Particles`, `RungeKutta2`)
- **Functions**: snake_case with `!` for mutation (`advection!`, `move_particles!`)
- **Kernels**: suffix `_kernel!` (`backtrack_kernel!`); the launching wrapper keeps the
  plain name
- Grid shorthand used across the codebase: `xci` (cell centers), `xvi` (cell vertices),
  `di` (grid spacing), `ni` (grid size) — reuse these, don't invent synonyms

## Common Pitfalls

1. Type instability in kernels ruins GPU performance — check with a CPU run first
2. Editing `ext/` for algorithm changes — shared code lives in `src/`, picked up by all
   backends automatically
3. New exports forgotten in `common.jl` — every public function is exported there
4. Orphaned test files — `test/runtests.jl` has explicit file lists for the CPU and GPU
   paths; a new `test_*.jl` must be added to both (see `.claude/rules/testing-rules.md`)
5. Breaking occupancy invariants — injection/cleanup/move logic is coupled; never change
   one without the others
6. Hardcoded `Float64` literals in kernels — breaks Metal
7. Commented-out code: delete it; git is the journal
8. Scope creep in PRs — unrelated cleanup goes in a separate PR

## Git Workflow

Follow [ColPrac](https://github.com/SciML/ColPrac). Feature branches, descriptive commits,
update tests and docs with code changes. Format with `git runic main` before pushing — CI
posts a diff comment on PRs that need formatting.

## CI Overview

- **GitHub Actions** `UnitTests.yml` — CPU tests, Julia 1.10/1.11/1 × Linux/Windows/macOS
- **Buildkite** (`.buildkite/`) — CUDA + AMDGPU GPU tests; the pipeline is regenerated by
  the `forerunner` plugin watching `run_tests.yml`
- `ci/cscs-*.yml` — CSCS cluster GPU CI
- `Format.yml` (Runic diff comment), `Downstream.yml` (downstream-package compatibility),
  `Documenter.yml` (docs build & deploy)

## Further Reading

Detailed reference docs are in `.agents/` — read on demand:

| Document | Content |
|----------|---------|
| `.agents/testing.md` | Running/writing tests, backend selection, debugging |
| `.agents/documentation.md` | Building docs, docstring conventions |
| `.agents/validation.md` | Validating advection/interpolation changes physically |

### Auto-loading Rules

Rules in `.claude/rules/` load automatically when you touch matching files:
- `kernel-rules.md` — KA kernel requirements (src/, ext/)
- `style-rules.md` — naming, comments, Runic (src/, test/, ext/)
- `testing-rules.md` — test writing and running (test/)
- `docs-rules.md` — documentation building and style (docs/)

### Skills (slash commands)

- `/run-tests` — run targeted tests, prioritized by what's likely to break
- `/build-docs` — build documentation locally
- `/add-feature` — checklist for adding new functionality
- `/new-miniapp` — set up and validate a new advection miniapp/benchmark script
- `/babysit-ci` — monitor CI, auto-fix small issues, pause on bigger problems
