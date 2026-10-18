# Input validation and failure behavior

Every public entry point checks its arguments on the host before it launches a kernel
that writes user data. A check that fails throws an `ArgumentError` or a
`DimensionMismatch` that names the offending argument, so the input data are left
unchanged. This page lists the policies these checks enforce.

## Precision, backend, and shape

- **Precision.** Every floating-point array passed together with a particle container
  must have the same scalar element type as the particle coordinates. Mixed precision is
  rejected, not converted, because a narrowing conversion inside a kernel is silent.
  Scalars such as `dt` are converted to the particle precision.
- **Metal and Float64.** Metal has no `Float64`. Constructors such as `init_particles`,
  `PhaseRatios`, `init_passive_markers`, `init_markerchain`, and `init_marker_surface`
  throw on the Metal backend when the grids or the requested type are `Float64`.
- **Backend.** Every array argument must live on the backend of the particles. Coordinate
  ranges carry no device storage and are accepted on every backend.
- **Grid fields.** Grid fields must have the size of the grid they are sampled on: the
  vertex grid for `particle2grid!`/`grid2particle!`, the cell centers for
  `particle2centroid!`/`centroid2particle!`. Along a direction with `ghost_i = true` (the
  default) the field includes the ghost nodes; along a direction with `ghost_i = false`
  it holds only the physical nodes.
- **Particle fields.** Per-particle fields must be `CellArray`s with the cell layout of
  `particles.index`, that is, the same grid size and the same number of slots per cell.
  Fields that hold `NaN` in empty slots (`args` of `move_particles!` and the injection
  routines) must be floating point.
- **Tuples.** A destination and its source must both be single fields or tuples of the
  same length.
- **Velocity.** `V` must be a tuple with one component per dimension, and `V[d]` must
  have the size of the staggered grid `grid_vi[d]`.

## Grids

A coordinate vector must hold at least two finite, strictly increasing entries, and all
grid coordinates must share one floating-point type. For `init_particles`, the
off-diagonal staggered grids `grid_vi[d][i]` are the cell-center grids of direction `i`
extended by one ghost center on each side, so they have one more entry than the vertex
grid `grid_vi[i][i]`.

## Aliasing

Source and destination arrays must not share memory. The checks reject:

- the particle coordinates among the companion fields `args` of `move_particles!` and
  the injection routines, and any two entries of `args` that share memory;
- a grid field and a particle field of an interpolation call that share memory, and
  entries of a tuple-valued destination that share memory;
- `F` and `F0` of `semilagrangian_advection!` that share memory;
- the grid field, marker field, and buffer of a passive-marker scatter that share
  memory.

## Coordinates and displacement

- **Material particles.** `move_particles!` rejects `NaN` in any active coordinate slot,
  because `NaN` means an invalid velocity field or time step. An infinite coordinate is
  valid input: it marks a particle that left the velocity grid during advection, and
  `move_particles!` removes it like any other particle outside the domain.
- **Boundary escape.** Particles that leave a non-periodic direction are discarded.
  Along a periodic direction they wrap to the opposite side; the ghost cells of a
  periodic direction must be empty when `move_particles!` is called.
- **Displacement.** A particle may cross any number of cells in one call to
  `move_particles!`. Large jumps are slower, not rejected; see the
  [`move_particles!`](@ref) docstring. If a destination cell is still full once no
  particle can move any more, the particle and its companion fields are dropped;
  `verbose = true` prints the number of dropped particles.
- **Passive markers.** Passive markers have no occupancy mask. `init_passive_markers`
  requires finite coordinates, and interpolation and advection require every marker to
  lie inside the grid. `advection!` stops markers on the boundary instead of moving them
  outside.
- **Time step and integrator.** `dt` must be finite. The advection routines accept
  `Euler`, `RungeKutta2`, and `RungeKutta4`; any other integrator is rejected.

## Empty support

A grid node whose interpolation stencil contains no particle has no data. JustPIC marks
such a node with `NaN` instead of inventing a value. This applies to:

- `particle2grid!` and `particle2centroid!` for material particles;
- `particle2grid!` for passive markers;
- `update_phase_ratios!`, `phase_ratios_center!`, `phase_ratios_vertex!`, and the
  face and midpoint phase ratios.

Test for `NaN` with `isnan` to detect empty support. Keep particle occupancy high enough,
for example with `inject_particles!`, to avoid it.

## Phase labels

A phase label must be an integer value in `1:nphases`, where `nphases` is the number of
phases of the `PhaseRatios` object. Phase labels may be stored with a floating-point
element type. The phase-ratio routines count invalid labels over the active slots and
throw if there is any.

## MPI and ghost layout

- `update_cell_halo!` requires an initialized ImplicitGlobalGrid.
- Passive markers do not support MPI runs with more than one rank.
- Grid fields follow the ghost-node layout described in [Grid layouts](grid_layouts.md).

## Unsupported combinations

The [support matrix](support_matrix.md) lists the combinations marked `Unsupported`. The
entry points reject these with an error instead of running an incorrect method:

- `Float64` on Metal, for every constructor;
- passive markers in MPI runs with more than one rank;
- advection integrators other than `Euler`, `RungeKutta2`, and `RungeKutta4`;
- 2D or 3D mismatches between the phase ratios and the particles, and 3D-only midpoint
  phase ratios requested for 2D particles.
