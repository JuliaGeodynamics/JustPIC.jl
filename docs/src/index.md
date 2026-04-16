# JustPIC.jl

JustPIC.jl provides Particle-in-Cell data structures and advection kernels for
2D and 3D geodynamic workflows on CPUs and accelerator backends.

## What the package covers

- Cell-sorted particle containers for efficient particle management.
- Particle, passive-marker, and marker-chain advection.
- Grid/particle transfer operators for nodal, staggered, and centroid-based data.
- Phase-ratio reconstruction and subgrid diffusion helpers.
- JLD2 checkpointing utilities for restart workflows.

## Getting started

The most common workflow is:

1. Pick a backend such as `CPUBackend`, `CUDABackend`, or `AMDGPUBackend`.
2. Build staggered velocity grids.
3. Initialize a particle container with `init_particles`.
4. Advect particles with `advection!`.
5. Restore cell locality with `move_particles!`.
6. Transfer data between grid and particles with `grid2particle!` or `particle2grid!`.

```julia
using JustPIC, JustPIC._2D

const backend = CPUBackend

xv = LinRange(0.0, 1.0, 33)
yv = LinRange(0.0, 1.0, 33)
xc = LinRange(step(xv) / 2, 1.0 - step(xv) / 2, 32)
yc = LinRange(step(yv) / 2, 1.0 - step(yv) / 2, 32)

grid_vx = xv, LinRange(first(yc) - step(yv), last(yc) + step(yv), length(yc) + 2)
grid_vy = LinRange(first(xc) - step(xv), last(xc) + step(xv), length(xc) + 2), yv

particles = init_particles(backend, 24, 48, 12, grid_vx, grid_vy)
method = RungeKutta2()

Vx = zeros(length.(grid_vx)...)
Vy = zeros(length.(grid_vy)...)

advection!(particles, method, (Vx, Vy), 1.0)
move_particles!(particles, ())
```

## Where to go next

- [Particles](particles.md): particle containers, initialization, and maintenance.
- [Interpolations](interpolations.md): grid/particle transfers and centroid helpers.
- [Marker chain](marker_chain.md): interface tracking with `MarkerChain`.
- [I/O](IO.md): checkpointing and restart.
- [Public API](API.md): compact reference of exported entry points.
