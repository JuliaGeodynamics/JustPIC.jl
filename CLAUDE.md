# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

JustPIC.jl is a Julia Particle-in-Cell (PIC) advection library for geodynamic simulations, running on CPU (multithreaded), Nvidia (CUDA) and AMD (AMDGPU) GPUs from the same source. It provides particle containers, advection integrators, particle↔grid interpolation, phase-ratio bookkeeping, marker chains (free-surface tracking) and passive tracers.

## Commands

Run from the repo root (`julia --project=.`), or from `test/` for test-only deps.

- Run full test suite (CPU): `julia --project -e 'using Pkg; Pkg.test()'`
- Run a single test file directly (fast iteration loop): `JULIA_JUSTPIC_BACKEND=CPU julia --project=test test/test_2D.jl` (swap in `test_3D.jl`, `test_integrators.jl`, `test_CellArrays.jl`, `test_save_load.jl`, `test_Aqua.jl`, `test_interpolation_kernels.jl`)
- Run against a GPU backend: `julia --project -e 'using Pkg; Pkg.test("JustPIC"; test_args=["--backend=CUDA"])'` (or `--backend=AMDGPU`); this `Pkg.add`s the GPU package on the fly and spawns one subprocess per test file whose name contains "2D"/"3D"
- Format check/fix: this repo uses **Runic**, not JuliaFormatter — `git runic main` (see `.github/workflows/Format.yml`); CI posts a diff comment on PRs that need formatting
- Build docs locally: `julia --project=docs docs/make.jl` (Documenter)

## Architecture

### Backend dispatch via textual `include`, not runtime branching

The core trick that explains most of the file layout: **the same implementation files are `include`d multiple times, once per backend, inside different modules that each set up their own `@init_parallel_stencil` context.**

- [src/JustPIC.jl](src/JustPIC.jl) is the top module. It defines backend tag types (`CPUBackend`, `AMDGPUBackend`; `CUDABackend` comes from the CUDA weakdep), the shared struct definitions used by every backend (`Particles`, `MarkerChain`, `PassiveMarkers`, `PhaseRatios` in [src/particles.jl](src/particles.jl) and [src/PhaseRatios/PhaseRatios.jl](src/PhaseRatios/PhaseRatios.jl)), and the advection integrator types (`Euler`, `RungeKutta2`, `RungeKutta4` in [src/Advection/types.jl](src/Advection/types.jl)).
- [src/JustPIC_CPU.jl](src/JustPIC_CPU.jl) defines `module _2D` and `module _3D`, each calling `@init_parallel_stencil(Threads, Float64, 2|3)` and then `include("common.jl")`.
- [src/common.jl](src/common.jl) is the actual algorithm catalog — it just chains `include`s for every feature area (CellArrays, Advection, Interpolations, Physics, Particles, MarkerChain, PassiveMarkers, PhaseRatios, IO) and re-exports their public functions. It contains no module boilerplate itself, which is what makes it re-includable.
- [ext/JustPICCUDAExt.jl](ext/JustPICCUDAExt.jl) and `ext/JustPICAMDGPUExt.jl` (loaded automatically when CUDA/AMDGPU are loaded, via `[weakdeps]`/`[extensions]` in [Project.toml](Project.toml)) each define their **own** `module _2D`/`module _3D` with `@init_parallel_stencil(CUDA|AMDGPU, ...)`, `include(".../src/common.jl")` again, then add a layer of thin `JustPIC._2D.f(...) = f(...)` dispatches constrained to `Particles{CUDABackend}`/`PassiveMarkers{CUDABackend}` etc., so the generic kernels defined in `common.jl` get compiled against the right array/kernel backend.
- `CA(backend, dims)` (cell-array constructor) and `TA(backend)` (plain array type, e.g. `Array`/`CuArray`/`ROCArray`) are the two functions every backend module overrides — look here first when adding backend-specific behavior.

When editing shared algorithm code, edit the file under `src/` (e.g. `src/Particles/move_safe.jl`) — it is picked up by all three backends automatically. Only touch the `ext/*.jl` files when you need backend-specific dispatch, type conversion (host↔device, e.g. the `CUDA.CuArray(::JustPIC.Particles)` conversions), or a genuinely different kernel implementation.

### Particle storage: `CellArray`s

Particles are *not* stored as flat arrays; they live in `src/CellArrays/`, a cell-indexed layout where each grid cell owns a fixed-capacity slot of particles (`nxcell`/`min_xcell`/`max_xcell` control target/min/max occupancy). `Particles.index` is a per-cell boolean mask marking which slots are occupied. This locality (particles physically grouped by cell) is what `move_particles!` ([src/Particles/move_safe.jl](src/Particles/move_safe.jl)) restores after advection — read that file before touching injection/cleanup logic since occupancy invariants span injection, cleanup and move together.

### Feature areas (all under `src/`, mirrored 1:1 for 2D/3D)

- `Advection/` + `Particles/Advection/` — integrators (Euler/RK2/RK4) and the actual particle advection + semilagrangian backtracking drivers (plain, `_LinP`, `_MQS` variants correspond to different interpolation schemes used during backtracking)
- `Interpolations/` — particle→grid, grid→particle (incl. FLIP variant `grid2particle_flip!`), particle→centroid, and MQS (moving quadratic scheme)/ndlerp low-level kernels
- `MarkerChain/` — a specialized 1D-in-2D (or 2D-in-3D) chain of markers tracking a free surface/topography; has its own advection/backtracking/resample/area logic since it can't reuse the volumetric particle machinery
- `PassiveMarkers/` — lightweight tracer-only particles (coords only, no occupancy metadata), for advection/interpolation that doesn't feed back into the simulation
- `PhaseRatios/` — computes per-phase area/volume fractions at cell centers, vertices, staggered velocity nodes, and (3D only) edge midpoints, from particle phase labels
- `Physics/subgrid_diffusion.jl` — subgrid-scale diffusion correction for advected fields
- `IO/JLD2.jl` — `checkpointing_particles` (JLD2-based particle checkpoint/restart)

### Test layout

`test/runtests.jl` is a custom runner (not plain `Test`): it reads `--backend=` (default `CPU`), sets `ENV["JULIA_JUSTPIC_BACKEND"]`, and either `include`s CPU test files in-process, or spawns a fresh `julia` subprocess per GPU test file (files are matched by `"2D"`/`"3D"` substring in the filename, not by directory). `test_Aqua.jl` runs Aqua.jl package-hygiene checks (ambiguities, stale deps, undefined exports) — two `_grid2particle` ambiguities are deliberately excluded there.

### CI

- GitHub Actions `UnitTests.yml` — CPU tests across Julia 1.10/1.11/1 × Linux/Windows/macOS
- Buildkite (`.buildkite/`) — CUDA and AMDGPU GPU tests, dynamically regenerated pipeline (`forerunner` plugin watches `run_tests.yml`)
- `ci/cscs-*.yml` — CSCS cluster GPU CI
- Formatting (Runic) and downstream-package compatibility checks also run as separate workflows
