# JustPIC.jl: package context and contributor guide

## Purpose and source of truth

JustPIC.jl provides Particle-in-Cell (PIC) advection and particle/grid coupling
for 2D and 3D geodynamics simulations. Material properties travel on particles;
Eulerian fields supply velocities and receive reconstructed properties. The
package also provides passive tracers, a 2D free-surface marker chain,
semi-Lagrangian grid-field advection, phase fractions, and subgrid thermal
diffusion. The calling application supplies the flow/physics solver, timestep,
boundary conditions, and simulation loop.

Use the current source and registered tests to resolve discrepancies in prose,
examples, or migration notes. `src/JustPIC.jl` and `src/common.jl` define what is
actually loaded. In particular, `src/Particles/move_safe.jl` is active;
`src/Particles/move.jl` is not included. Use `using JustPIC`; algorithms are
included directly into one module, not separate 2D/3D modules.

`Project.toml` is authoritative for versions and dependencies. Julia 1.10+ is
required within its declared compatibility bounds. Main dependencies are
KernelAbstractions, CellArrays, CellArraysIndexing, StaticArrays,
GridGeometryUtils, ImplicitGlobalGrid, MPI, JLD2, MuladdMacro, and Statistics.
CUDA, AMDGPU, and Metal are weak dependencies that activate package extensions.

## Architecture and where to look

| Location | Responsibility |
| --- | --- |
| `src/JustPIC.jl` | Module imports, base allocation dispatch, core includes and exports. |
| `src/common.jl` | Active shared algorithm include graph and most public exports. |
| `src/particles.jl` | `AbstractParticles`, `Particles`, `MarkerChain`, `PassiveMarkers`, geometry/access helpers. |
| `src/launch.jl` | Backend discovery, synchronous kernel launching, CellArray allocation, device/precision conversion of grids. |
| `src/Utils.jl` | Ghost coordinates, interior sizes/masks, spacing access and other grid utilities. |
| `src/CellArrays/` | CellArray helpers, host/container copying and conversion, MPI halo exchange. |
| `src/Advection/` | Integrator types and common integration/advection methods. |
| `src/Particles/particles_utils.jl` | Particle initialization and companion field allocation. |
| `src/Particles/utils.jl` | Particle geometry, coordinate extraction, cell lookup and related helpers. |
| `src/Particles/move_safe.jl` | Cell reassignment, occupancy/deletion helpers, periodic wrapping and cleanup. |
| `src/Particles/injection.jl`, `forced_injection.jl` | Replenishment of depleted cells and explicit insertion. |
| `src/Particles/Advection/` | Particle Euler/RK2/RK4, velocity interpolation variants, grid-field backtracking. |
| `src/Interpolations/` | Grid/particle and centroid transfers, multilinear interpolation and MQS. |
| `src/PhaseRatios/` | Phase storage, constructors and reconstruction at centers, vertices and staggered locations. |
| `src/Physics/subgrid_diffusion.jl` | Particle temperature relaxation and grid correction with reusable scratch arrays. |
| `src/MarkerChain/` | Surface initialization, column reassignment, resampling, topography, rock fractions and advection. |
| `src/PassiveMarkers/` | Flat tracer initialization, advection and field transfers. |
| `src/IO/JLD2.jl` | Backend-independent particle checkpoint writing. |
| `ext/JustPIC{CUDA,AMDGPU,Metal}Ext.jl` | Vendor-specific allocations, constructors and conversions; shared algorithms stay in `src/`. |
| `test/` | Numerical/regression tests and explicit backend-aware runner. |
| `docs/src/`, `docs/examples/` | Documenter manual and example scripts. |
| `.agents/` | Focused testing, documentation and physical-validation guidance. |
| `.github/workflows/` | CI tests, documentation, Runic formatting and downstream checks. |

## Data model and grid conventions

### Material particles

`Particles{Backend,N,...}` stores one coordinate `CellArray` per dimension in
`coords`, plus a boolean `index` CellArray marking live slots. Each logical grid
cell has fixed storage capacity `max_xcell`. `nxcell` is the initialization target
and `min_xcell` drives replenishment. Random initialization rounds the target up
to a multiple of four quadrants in 2D or eight octants in 3D.

The `np` field is allocated slot capacity in the current constructor, **not the
live particle count**. Use the occupancy mask (`sum(particles.index.data)`,
accounting for halos when appropriate). Empty coordinate slots contain `NaN`;
NaNs in inactive slots are expected and must not be counted as numerical failure.
Do not assume active slots form a contiguous prefix.

Particle fields such as temperature and phase labels are separate CellArrays.
`init_cell_arrays(particles, Val(N))` returns an N-tuple with matching layout and
precision. Pass every persistent companion field in the `args` tuple when moving
particles so values remain attached to their coordinates.

CellArray logical cell indexing and slot indexing differ from backing-array
indexing. Use `cellaxes`, `cellnum`, `@cell`, and `CAI.@index A[ip, I...]` as in
nearby code; do not infer a portable layout from `A.data`. CPU allocations use
block length 1, while GPU allocations use the backend's structure-of-arrays layout.

### Geometry and ghosts

`init_particles(backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy[, grid_vz])`
takes a tuple of coordinate vectors for each staggered velocity component. In 2D,
the pattern is `(xv, yc_extended)` for Vx and `(xc_extended, yv)` for Vy.
`particles.xi_vel` stores those grids. `xvi` and `xci` store vertex and center
coordinates extended by one periodic ghost node at each end. Creating ghost
coordinates does not itself enable periodic particle motion or populate field halos.

`di` and `_di` hold spacings and inverse spacings under `center`, `vertex`, and
`velocity`. Uniform-range paths use scalar spacings; refined-vector paths use
local spacing arrays. Use `@dxi`/grid helpers instead of assuming one global `dx`.
GridGeometryUtils supplies geometry operations used by these paths.

Interior kernels commonly launch over sizes minus two and shift indices by one.
Vertex fields normally match `length.(particles.xvi)` and centroid fields match
`length.(particles.xci)`. Several transfer APIs accept per-axis `ghost_1`,
`ghost_2`, `ghost_3` flags for physical-only arrays. Check the exact overload and
its layout checks before changing shapes or offsets.

## How a PIC step works

After constructing staggered grids, particles and a grid field `T`, the core
scalar-field workflow is:

```julia
particle_args = (pT,) = init_cell_arrays(particles, Val(1))
grid2particle!(pT, T, particles)
for it in 1:nsteps
    advection!(particles, RungeKutta2(), V, dt)
    move_particles!(particles, particle_args)
    inject_particles!(particles, particle_args)
    particle2grid!(T, pT, particles)
end
```

- `advection!` updates coordinates using a staggered velocity field; it does not
  restore particle ownership by cell. Euler, RK2 and RK4 are integrator choices;
  `advection_LinP!` and `advection_MQS!` select alternative velocity interpolation.
- `move_particles!` reassigns coordinates and companion fields together. Its
  current implementation launches 3^N cell-color passes. Preserve its write
  ordering and inspect collision behavior before changing destination searches,
  concurrency or timestep assumptions.
- The physical domain is half-open. Outflow through nonperiodic boundaries is
  discarded; `periodic_1/2/3` enable wrapping. A particle can also be lost when
  its destination has no free slot: unconditional count conservation is not a
  guarantee of the current implementation.
- `inject_particles!` replenishes underpopulated quadrants/octants, with field
  initialization handled by the matching overload. `inject_particles_phase!`
  handles phase-aware workflows. `force_injection!` inserts explicit coordinates
  and supplied values into free slots. Inspect these together with movement when
  changing occupancy behavior.
- `clean_particles!` removes active particles outside their owning cells and
  clears associated storage; do not assume it compacts slots just because an old
  docstring describes compaction.

The example is the coupling sequence, not a complete physics solver. The caller
chooses `dt`, updates velocities/resolved fields, and manages boundary conditions.

## Other algorithms and coupled state

- **Field transfers:** `grid2particle!` samples vertex fields;
  `particle2grid!` reconstructs vertex fields from weighted particle contributions.
  `centroid2particle!` and `particle2centroid!` use cell centers.
  `grid2particle_flip!` combines incremental and direct transfers; inspect its
  alpha convention before modifying it. Scalar and tuple overloads exist.
- **Semi-Lagrangian fields:** `semilagrangian_advection!` and its LinP/MQS
  variants backtrack destination grid nodes through velocity, then sample the
  old field `F0` into `F`. This is a grid-field operation, distinct from moving
  the material-particle container. Preserve source/destination separation.
- **Phase ratios:** per-particle labels are converted into fractional occupancy
  at `center`, `vertex`, `Vx`, `Vy`, and in 3D `Vz`, `yz`, `xz`, `xy` locations.
  The 2D container uses small dummy arrays for unused fields to maintain concrete
  types. `nphases` returns a `Val`; `numphases` returns an integer. Validate
  normalization and empty-neighborhood behavior when changing reconstruction.
- **Subgrid diffusion:** `SubgridDiffusionCellArrays` stores old temperatures,
  particle increments, local timescales and a grid correction buffer.
  `subgrid_diffusion!` and `subgrid_diffusion_centroid!` relax particle
  temperatures toward the grid and apply the corresponding correction. Match
  scratch-buffer location (`:vertex` or `:center`) and ghost layout to the API.
- **Marker chain:** a 2D single-valued surface `y = h(x)`, with markers bucketed
  by horizontal column. State includes current/previous coordinates and vertex
  heights. Vertices must be finite and strictly increasing; refined spacing is
  supported through local column widths. `cell_length(chain, i)` is local;
  `cell_length(chain)` requires uniform spacing. `advect_markerchain!` performs
  advection, movement, resampling, topography reconstruction, mean-height
  correction and old-state refresh. Plain `advection!` only moves coordinates.
  Backtracking and `compute_rock_fraction!` provide related surface operations.
- **Passive markers:** coordinate arrays and a count, without material-particle
  cell occupancy. They have their own advection and transfer overloads; do not
  assume all `AbstractParticles` subtypes have `index`, `xvi`, or `di` fields.
- **MPI:** `update_cell_halo!` stages each CellArray slot through an ordinary
  backend array for ImplicitGlobalGrid halo exchange. Exchange coordinates,
  occupancy and companion fields before reassignment across ranks; refresh halos
  after injection before reconstructing grid fields. Read
  `docs/src/field_advection2D_MPI.md` for the distributed workflow.
- **Checkpointing:** `checkpointing_particles` converts containers/arrays to
  host representations, writes JLD2 through a temporary file, then replaces the
  destination. It stores particles and optional phases, ratios, chain, time,
  timestep, particle fields and extra keywords. Rank-local names use a four-digit
  rank suffix. See `test/test_save_load.jl` for reload and conversion patterns.

## Backend and implementation rules

- Use KernelAbstractions backend types for construction: `JustPIC.CPU`,
  `CUDA.CUDABackend`, `AMDGPU.ROCBackend`, `Metal.MetalBackend`. Load the vendor
  package to activate its extension. `TA(backend)` selects plain array storage;
  `cell_array`/`CA` allocate CellArrays.
- `ka_backend(x)` returns the backend instance used for execution. Shared kernels
  use `@kernel` and KernelAbstractions `@index`; CellArraysIndexing access must be
  qualified as `CAI.@index` to avoid the macro-name collision.
- Launch through `launch!(ka_backend(x), kernel!, ndrange, args...)`. It currently
  synchronizes after every launch. Host reads, halo exchanges and successive
  mutation passes depend on this ordering; changing it is an architectural change.
- Keep shared algorithms in `src/`; vendor-specific allocation/conversion belongs
  in `ext/`. Do not branch on `Array`, `CuArray` or other vendor array types inside
  shared algorithms. Inputs must use compatible backends; selecting the backend
  from a particle container does not automatically transfer every input.
- Kernels must be allocation-free and type-stable. Prefer tuples, `SVector`s,
  concrete types and inline helpers over temporary dynamic arrays.
- Keep numeric precision generic. Metal has no Float64 support: use typed
  constants, `zero`/`one`, and conversions to the field/coordinate type.
  `set_precision`, `recast_grid`, `device_grid`, and `backend_grid` handle
  integrator/grid adaptation. Even nominally Float32 Base ranges can introduce
  Float64 internals, which is why range recasting exists.
- Before changing movement, cleanup or injection, read
  `src/Particles/move_safe.jl` and preserve coordinate/mask/field consistency.
- Do not change dependencies or compatibility bounds unless required by the task.
  Add public exports in the appropriate module export list (normally
  `src/common.jl`) and document new APIs in `docs/src/API.md`.

## Tests, documentation and validation

Read `.agents/testing.md` for test changes, `.agents/documentation.md` for manual
changes, and `.agents/validation.md` for advection/interpolation changes. Treat
their examples as guidance and check current code: their test-file lists can lag
behind `test/runtests.jl`, and conservation/NaN checks must account for inactive
slots, halos, outflow and capacity losses.

```sh
# Full CPU suite
julia --project=. -e 'using Pkg; Pkg.test()'

# GPU suite: substitute AMDGPU or Metal as needed
julia --project=. -e 'using Pkg; Pkg.test(; test_args=["--backend=CUDA"])'

# Manual (instantiate the docs environment first if needed)
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

`test/runtests.jl` explicitly selects files; new files are not automatically run.
The CPU path includes tests in-process. GPU tests run in fresh subprocesses with
the Pkg.test sandbox project and a load path containing this repository. Preserve
that environment forwarding. GPU extras come from root `Project.toml`; there is
no separate `test/Project.toml`. Tests select their backend with
`JULIA_JUSTPIC_BACKEND`. For individual files, use an environment containing test
extras (the TestEnv workflow is documented in `.agents/testing.md`).

| Test | Main area |
| --- | --- |
| `test_2D.jl`, `test_3D.jl` | Dimension-specific PIC operations and numerical miniapps. |
| `test_integrators.jl` | Integrator behavior. |
| `test_interpolation_kernels.jl` | Interpolation and related kernel regressions. |
| `test_refined_grid.jl` | Nonuniform grid behavior. |
| `test_markerchain_2D.jl` | Surface geometry, advection and resampling. |
| `test_CellArrays.jl` | Cell storage and conversions. |
| `test_save_load.jl` | Checkpoint round-trips. |
| `test_Aqua.jl` | Package hygiene and migration checks. |

Debug on CPU first, then validate affected GPU paths when hardware is available.
New tests should use the file's backend selection and move device data to host
for inspection instead of enabling scalar indexing. Use small grids and test
numerical behavior: analytic trajectories/convergence, linear-field transfers,
phase normalization, locality, and expected particle counts. Surface changes
also need bounded rock fractions, ordered reconstructed geometry and appropriate
height conservation. Longer advection changes may need short physical miniapps
to expose clumping, drift or diffusion beyond unit-test tolerances.

The manual entry point is `docs/make.jl`; it adds the repository to `LOAD_PATH`.
Relevant pages include `particles.md`, `CellArrays.md`, `interpolations.md`,
`velocity_interpolation.md`, `marker_chain.md`, `mixed_CPU_GPU.md`, and `IO.md`.
Update documentation with behavior changes. Exported docstrings should include
a signature, concise purpose, arguments and important layout constraints.

## Working practices

Preserve unrelated working-tree changes and local experiments. Follow
`CONTRIBUTING.md`; keep patches focused. Use Runic for Julia formatting, not
JuliaFormatter, and format changed Julia files before opening a PR. Register new
tests and run the most relevant CPU checks before broadening validation. Report
what actually ran and any unavailable backend checks. Documentation-only changes
need appropriate prose/link checks, not numerical tests by default.

Keep this guide aligned with architecture changes, especially module includes,
storage invariants, grid/ghost conventions, backend contracts and test entry points.
