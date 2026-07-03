---
name: new-miniapp
description: Set up and validate a new JustPIC advection miniapp or benchmark script
---

# New Miniapp

Set up, run, and validate a script that exercises JustPIC on a concrete flow — for
demonstration, benchmarking, or validating a feature.

## Step 1: Understand the Case

Clarify with the user (or infer from the request):
- Dimension (2D/3D), grid size, domain extents
- Velocity field: analytic (solid-body rotation, pure shear, cellular flow) or loaded
- What is advected: volumetric `Particles`, a `MarkerChain` (free surface), or
  `PassiveMarkers`
- Integrator (`RungeKutta2` is the default choice) and what to measure/visualize

## Step 2: Skeleton

Follow the existing patterns (see the Miniapps testsets in `test/test_2D.jl` and the
scripts in `docs/examples/`):

```julia
using JustPIC
# grid: cell vertices xv/yv, cell centers xc/yc; velocity nodes are staggered
grid_vx = xv, expand_range(yc)     # Vx nodes: vertices in x, extended centers in y
grid_vy = expand_range(xc), yv
nxcell, max_xcell, min_xcell = 24, 48, 14
particles = init_particles(backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy)
# velocity arrays live on those staggered nodes; phases via init_cell_arrays / cell_array
```

Time loop core: `advection!` → `move_particles!` → (`inject_particles!` if needed) →
interpolation (`particle2grid!` / `grid2particle!`).

- `backend` is a KA backend type: `CPU`, or `CUDA.CUDABackend` after `using CUDA`
- Keep the script backend-agnostic: allocate through `TA(backend)` / `cell_array(backend, ...)`

## Step 3: Verify Setup Before Long Runs

- Print/plot the initial particle distribution; check per-cell counts equal `nxcell`
- `extrema` of the velocity field make physical sense
- One timestep: no NaNs (`any(isnan, particles.coords[1].data)`), particle count conserved

## Step 4: Short Run + Validation

Apply the ladder in `.agents/validation.md`:
- analytic check (e.g. blob returns after one rotation period)
- particle count / occupancy invariants each step
- CPU vs GPU parity if a GPU is available

## Step 5: Visualize

Use GLMakie/CairoMakie: scatter particle positions colored by phase (bring device data
back with `Array(...)` first), or heatmap the regridded field. For time series, record
frames each N steps.

## Output Location

- Throwaway experiments: keep out of the repo (scratch directory)
- Reusable examples: `docs/examples/`, referenced from a docs page
- Regression-worthy cases: distill into a testset in `test_2D.jl`/`test_3D.jl`

## Common Issues

- **NaNs**: `dt` too large, or backtracking leaves the domain — reduce `dt` first
- **Particle starvation/clumping**: injection not keeping up; check `min_xcell`,
  consider `force_injection!`
- **GPU mismatch vs CPU**: type instability or missing sync — reproduce on CPU
- **Velocity/particles on different backends**: conversions are the caller's job
