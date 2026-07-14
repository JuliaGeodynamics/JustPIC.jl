# JustPIC.jl

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
`AMDGPU`, or `Metal`) and select its backend — see [Mixed CPU/GPU](mixed_CPU_GPU.md).

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

- [Particles](particles.md) — the `Particles` container and its lifecycle
- [CellArrays](CellArrays.md) — the cell-local storage layout
- [Interpolations](interpolations.md) and [Velocity interpolation](velocity_interpolation.md)
- [Marker chain](marker_chain.md) — free-surface / topography tracking
- [I/O](IO.md) — checkpointing and restart
- [Mixed CPU/GPU](mixed_CPU_GPU.md)
- [Public API](API.md)

## Funding

The development of this package is supported by the
[GPU4GEO](https://ptsolvers.github.io/GPU4GEO/) [PASC](https://www.pasc-ch.org)
project.
