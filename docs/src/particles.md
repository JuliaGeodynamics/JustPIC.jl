# Particles

## Memory layout

`Particles` and `MarkerChain` store coordinates in `CellArray`s from
[`CellArrays.jl`](https://github.com/omlins/CellArrays.jl). Entries are grouped
by parent cell so that interpolation, advection, and reinjection routines work
on spatially local chunks of memory.

## Particle objects

JustPIC exposes three concrete `AbstractParticles` containers:

- `Particles`: the main cell-sorted container used for PIC workflows.
- `PassiveMarkers`: lightweight tracers stored as flat coordinate arrays.
- `MarkerChain`: a 2D interface-tracking container for free surfaces and similar boundaries.

`Particles` stores coordinates, active-slot masks, occupancy thresholds, and the
derived center, vertex, and velocity grids used by the high-level APIs. Those
extra fields are what let helpers such as `move_particles!`, `grid2particle!`,
`particle2grid!`, `inject_particles!`, `update_phase_ratios!`, and
`subgrid_diffusion!` use the compact `(..., particles, ...)` call style.

## Initialization

`init_particles` accepts either:

- a scalar `nxcell` for random, quadrant-balanced seeding inside each cell, or
- a tuple `nxcell` for a regular per-dimension layout.

After construction, the returned `Particles` object already contains the center,
vertex, and staggered velocity grids needed by the higher-level APIs.

### Randomly distributed particles

```julia
backend   = JustPIC.CPUBackend # device backend
nxcell    = 24  # initial number of randomly distributed particles
max_xcell = 48  # maximum number of particles per cell
min_xcell = 12  # minimum number of particles per cell
n         = 32  # number of cells per dimension
Lx   = Ly = 1.0 # domain size
xvi       = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n) # nodal vertices
dxi       = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
xci       = xc, yc = LinRange(dx / 2, Lx - dx / 2, n - 1), LinRange(dy / 2, Ly - dy / 2, n - 1)
grid_vx   = xv, LinRange(first(yc) - dy, last(yc) + dy, length(yc) + 2)
grid_vy   = LinRange(first(xc) - dx, last(xc) + dx, length(xc) + 2), yv
## initialize particles object with randomly distributed coordinates
particles = init_particles(
    backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy,
)
```

### Regularly spaced particles

```julia
backend   = JustPIC.CPUBackend # device backend
nxcell    = (5, 5)  # number of evenly spaced particles in the x- and y- dimensions
max_xcell = 48      # maximum number of particles per cell
min_xcell = 12      # minimum number of particles per cell
n         = 32      # number of cells per dimension
Lx   = Ly = 1.0     # domain size
xvi       = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n) # nodal vertices
dxi       = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
xci       = xc, yc = LinRange(dx / 2, Lx - dx / 2, n - 1), LinRange(dy / 2, Ly - dy / 2, n - 1)
grid_vx   = xv, LinRange(first(yc) - dy, last(yc) + dy, length(yc) + 2)
grid_vy   = LinRange(first(xc) - dx, last(xc) + dx, length(xc) + 2), yv
## initialize particles object with randomly distributed coordinates
particles = init_particles(
    backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy,
)
```

## Particle maintenance

These helpers keep the cell-sorted storage healthy as particles move:
`move_particles!`, `inject_particles!`, `inject_particles_phase!`,
`clean_particles!`, `force_injection!`, `cell_index`, and `cell_length`.

Once `particles` has been initialized, most transfer and maintenance routines
use the geometry stored in the container directly:

```julia
grid2particle!(Fp, F, particles)
particle2grid!(F, Fp, particles)
move_particles!(particles, particle_args)
inject_particles!(particles, particle_args)
update_phase_ratios!(phase_ratios, particles, phases)
```

This is the preferred high-level style for simulation code, tests, and examples.
