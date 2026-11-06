```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: JustPIC.jl
  text: Particle-in-Cell advection on CPUs and GPUs
  tagline: Particle containers, advection, and particle/grid interpolation for large-scale, multi-XPU geodynamics simulations.
  actions:
    - theme: brand
      text: Getting Started
      link: /particles
    - theme: alt
      text: API Reference
      link: /API
    - theme: alt
      text: Performance
      link: /performance
    - theme: alt
      text: View on GitHub
      link: https://github.com/JuliaGeodynamics/JustPIC.jl
  image:
    src: /logo.svg
    alt: JustPIC.jl

features:
  - title: Backend agnostic
    details: The same KernelAbstractions kernels run on CPU, CUDA, AMDGPU, and Metal; the backend is chosen by the array type.
    link: /mixed_CPU_GPU

  - title: Particles and interpolation
    details: Cell-local particle storage with injection, advection, and grid-to-particle and particle-to-grid interpolation.
    link: /particles

  - title: Free surfaces
    details: Marker chains in 2D and marker surfaces in 3D track topography advected by the flow.
    link: /marker_surface

  - title: Examples
    details: Runnable 2D, 3D, periodic, and MPI field-advection examples.
    link: /field_advection2D
---
```

## What is JustPIC.jl?

Particle-in-Cell (PIC) advection for large-scale, multi-XPU simulations. JustPIC
runs the same code on CPU, CUDA, AMDGPU, and Metal through
[KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl): the
compute backend is chosen by the array type, not by rewriting the algorithm.

## Installation

JustPIC is registered in the General registry:

```julia
using Pkg
Pkg.add("JustPIC")
```

To run on a GPU, additionally install the matching backend package (`CUDA`,
`AMDGPU`, or `Metal`) and select its backend — see [Backends](backends.md).

## Getting started

Choose a backend, build a staggered grid, and initialize a particle container:

```jldoctest
julia> using JustPIC

julia> backend = JustPIC.CPU;

julia> nxcell, max_xcell, min_xcell = 6, 12, 3;

julia> n = 8;  # vertices per dimension

julia> xv = yv = LinRange(0.0, 1.0, n);

julia> dx = dy = xv[2] - xv[1];

julia> xc = yc = LinRange(dx / 2, 1 - dx / 2, n - 1);

julia> grid_vx = xv, LinRange(first(yc) - dy, last(yc) + dy, length(yc) + 2);

julia> grid_vy = LinRange(first(xc) - dx, last(xc) + dx, length(xc) + 2), yv;

julia> particles = init_particles(backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy);

julia> sum(particles.index.data)  # number of active markers across the grid
392
```

`particles` can then be advected through a velocity field and used for
grid-to-particle and particle-to-grid interpolation. See [Particles](particles.md)
and [Interpolations](interpolations.md) for the full workflow, or the runnable
[2D field advection example](field_advection2D.md).

## Manual

- [Particles](particles.md) and [Particle lifecycle](particle_lifecycle.md) — the
  `Particles` container and the interpolate/advect/move/inject time step
- [Grid layouts](grid_layouts.md) — vertex, centroid, and staggered velocity grids
- [Backends](backends.md) — running on CPU, CUDA, AMDGPU, and Metal
- [CellArrays](CellArrays.md) — the cell-local storage layout
- [Interpolations](interpolations.md), [PIC/FLIP](pic_flip.md), and
  [Velocity interpolation](velocity_interpolation.md)
- [Particle fields](particle_fields.md) — per-particle data carried with the particles
- [Marker chain](marker_chain.md) — 2D free-surface / topography tracking
- [Marker surface](marker_surface.md) — 3D free-surface / topography tracking
- Examples: [2D](field_advection2D.md), [2D with MPI](field_advection2D_MPI.md),
  [3D](field_advection3D.md), [periodic](donut_advection_periodic.md), and
  [PIC/FLIP](temperature_flip.md) advection
- [I/O](IO.md) — checkpointing and restart
- [Mixed CPU/GPU](mixed_CPU_GPU.md) — particles on the CPU alongside a GPU solver
- [Input validation](validation.md) and [Support matrix](support_matrix.md)
- [Public API](API.md)
- [Performance](performance.md) — benchmark history on CPU and GPU backends

## Funding

The development of this package is supported by the
[GPU4GEO](https://ptsolvers.github.io/GPU4GEO/) [PASC](https://www.pasc-ch.org)
project.
