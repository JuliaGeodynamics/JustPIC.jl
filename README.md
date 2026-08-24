[![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliageodynamics.github.io/JustPIC.jl/stable/)
[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliageodynamics.github.io/JustPIC.jl/dev/)
[![Julia tests](https://github.com/JuliaGeodynamics/JustPIC.jl/actions/workflows/UnitTests.yml/badge.svg)](https://github.com/JuliaGeodynamics/JustPIC.jl/actions/workflows/UnitTests.yml)
[![GPU tests](https://badge.buildkite.com/bb05ed7ef3b43f843a5ba4a976c27a724064d67955193accea.svg?branch=main)](https://buildkite.com/julialang/justpic-dot-jl)
[![codecov](https://codecov.io/gh/JuliaGeodynamics/JustPIC.jl/graph/badge.svg?token=PN0AJZXK13)](https://codecov.io/gh/JuliaGeodynamics/JustPIC.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

# JustPIC.jl

JustPIC.jl is a backend-generic Julia library for Particle-in-Cell (PIC)
advection and particle/grid interpolation. It is designed for large-scale
geodynamics simulations and supports 2D and 3D workflows on CPUs, Nvidia GPUs,
AMD GPUs, and Apple GPUs through
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

The algorithm is selected by the array backend, so the same application code
can run on different hardware.

## Installation

JustPIC.jl is registered in Julia’s General registry:

```julia
using Pkg
Pkg.add("JustPIC")
```

The CPU backend is available immediately. For GPU execution, install the
matching optional package (`CUDA`, `AMDGPU`, or `Metal`) and select its
KernelAbstractions backend. See the [mixed CPU/GPU guide](https://juliageodynamics.github.io/JustPIC.jl/dev/mixed_CPU_GPU/).

## What it provides

- Cell-local particle storage with injection, cleanup, and locality-preserving
  movement.
- Euler, RK2, and RK4 particle advection, plus linear, MQS, and
  semi-Lagrangian variants.
- Grid-to-particle, particle-to-grid, and particle-to-centroid interpolation.
- Phase-ratio updates, subgrid diffusion, and passive markers.
- Marker chains for free-surface and topography tracking, including advection
  and resampling.
- JLD2 checkpointing and MPI-aware cell-halo updates.

## Minimal example

```julia
using JustPIC

backend = JustPIC.CPU
n = 33
L = 1.0
xv = yv = LinRange(0.0, L, n)
dx = dy = xv[2] - xv[1]
xc = yc = LinRange(dx / 2, L - dx / 2, n - 1)

# Staggered velocity-grid coordinates, including one ghost layer.
grid_vx = xv, LinRange(first(yc) - dy, last(yc) + dy, length(yc) + 2)
grid_vy = LinRange(first(xc) - dx, last(xc) + dx, length(xc) + 2), yv

particles = init_particles(backend, 8, 16, 4, grid_vx, grid_vy)

vx(x, y) = 250 * sin(π * x) * cos(π * y)
vy(x, y) = -250 * cos(π * x) * sin(π * y)
V = (
    TA(backend)([vx(x, y) for x in grid_vx[1], y in grid_vx[2]]),
    TA(backend)([vy(x, y) for x in grid_vy[1], y in grid_vy[2]]),
)

dt = min(dx / maximum(abs, V[1]), dy / maximum(abs, V[2]))
scheme = RungeKutta2()

for _ in 1:100
    advection!(particles, scheme, V, dt)
    move_particles!(particles, ())
end
```

`TA(backend)` maps the KernelAbstractions backend to its plain array type:
`Array` on the CPU and the corresponding device array type when a GPU extension
is loaded. For a complete workflow, including interpolation and visualization,
see the [documentation](https://juliageodynamics.github.io/JustPIC.jl/dev/).

## Development

Run the CPU test suite from a checkout with:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Build the documentation locally with:

```sh
julia --project=docs docs/make.jl
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution requirements.

## Funding

Development is supported by the [GPU4GEO](https://ptsolvers.github.io/GPU4GEO/)
[PASC](https://www.pasc-ch.org) project.
