# Marker chain

In geodynamic modeling it is often useful to track interfaces between different materials,
such as a topographic profile or a lithological boundary. JustPIC.jl provides the
`MarkerChain` object to define and advect **open surfaces** (not closed polygons) in
two-dimensional models. A marker chain represents a single-valued height field `y = h(x)`:
markers are distributed along a horizontal grid and can only describe surfaces that do not
fold back on themselves.

## The `MarkerChain` object

A `MarkerChain` stores its markers in the same column-wise `CellArray` layout used by
[`Particles`](particles.md): the horizontal grid `cell_vertices` (`xv`) divides the domain
into columns, and each column holds up to `max_xcell` marker slots guarded by a boolean
occupancy mask. Alongside the markers, the chain caches the topography sampled at the grid
vertices (`h_vertices`). The most useful fields are:

| Field | Meaning |
|-------|---------|
| `coords` | marker coordinates, `coords[1] = x`, `coords[2] = y` (empty slots are `NaN`) |
| `coords0` | marker coordinates from the previous time step |
| `h_vertices` | topography at the grid vertices (current step) |
| `h_vertices0` | vertex topography from the previous step (used for mass conservation) |
| `cell_vertices` | the horizontal grid `xv` defining the columns |
| `index` | per-slot occupancy mask |
| `min_xcell` / `max_xcell` | minimum / maximum markers allowed per column |

Marker precision follows the grid/elevation element type, so a `Float32` grid produces a
`Float32` chain — this is required on Metal, which has no `Float64`.

## Creating a chain

Instantiate a chain with a constant elevation `h`:

```julia
chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, h)
```

where `backend` is the device backend, `nxcell` is the initial number of markers per column,
`min_xcell`/`max_xcell` are the minimum/maximum markers allowed per column, and `xv` is the
grid of vertices. `h` may also be a vector giving the initial elevation at each vertex.

You can overwrite the geometry of an existing chain in two ways. From an arbitrary
topographic polyline:

```julia
# create a topographic profile spanning the chain's horizontal extent
topo_x = LinRange(0, 1, 200)
topo_y = @. sin(2π * topo_x) * 0.1

fill_chain_from_chain!(chain, topo_x, topo_y)
```

or directly from heights defined at the grid vertices:

```julia
fill_chain_from_vertices!(chain, topo_y)   # length(topo_y) == length(xv)
```

Both refresh the cached vertex topography and the previous-step buffers so the chain is
ready to advect.

## Advecting a chain

JustPIC offers two schemes to evolve a chain in a velocity field `V` (a tuple of the
velocity component arrays) defined on the staggered grids `grid_vxi`.

**Lagrangian** — move the markers themselves, then reassign, resample, and rebuild the
topography:

```julia
advect_markerchain!(chain, method, V, grid_vxi, dt)
```

**Semi-Lagrangian** — backtrack the vertex heights through the velocity field, limit steep
slopes, conserve mass, and reconstruct the markers:

```julia
semilagrangian_advection_markerchain!(chain, method, V, grid_vxi, xvi, dt; max_slope_angle = 45.0)
```

where `xvi = (xv, yv)` is the chain's vertex grid. The semi-Lagrangian scheme is more robust
on steep or strongly sheared surfaces where marker advection would tangle. Both schemes
conserve the mean height. `method` is the time integrator: `advect_markerchain!` accepts
`Euler`, `RungeKutta2`, or `RungeKutta4`, while the semi-Lagrangian scheme requires a
backtracking-capable integrator (`RungeKutta2` or `RungeKutta4`).

## Resampling and topography

As a chain deforms, columns can become depleted or unevenly spaced. `resample!` refills any
column that drops below `min_xcell` markers (or becomes too distorted) and restores a regular
spacing:

```julia
resample!(chain)
```

`advect_markerchain!` calls this internally, so you only need it when composing the advection
steps yourself. To synchronize the cached vertex topography with the current marker positions
use `compute_topography_vertex!(chain)`.

## Coupling to a two-phase Stokes solver

To use a marker chain as a free surface / air-rock interface, compute the fraction of each
control volume that lies below the chain with `compute_rock_fraction!`:

```julia
compute_rock_fraction!(ratios, chain, xvi, dxi)
```

`ratios` is a container with `center`, `vertex`, `Vx`, and `Vy` fields (e.g. a `PhaseRatios`
or a named tuple of arrays) sized to the respective grid locations; each is filled with a
value in `[0, 1]`. `xvi` is the vertex grid and `dxi` the grid spacing.

You can also interpolate the grid velocity onto the current marker positions (for diagnostics
or custom advection) with `interpolate_velocity_to_markerchain!(chain, chain_V, V, grid_vxi)`,
where `chain_V` is preallocated with the chain's cell layout.

## Example

```julia
using JustPIC
using GLMakie

const backend = JustPIC.CPU

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1 - dx; sigdigits = 5)
    xF = round(x2 + dx; sigdigits = 5)
    return LinRange(xI, xF, n + 2)
end

# velocity field
vi_stream(x) = π * 1e-5 * (x - 0.5)

# Initialize domain & grids
n = 51
Lx = Ly = 1.0
# nodal vertices
xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
# nodal centers
xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1)
# staggered grid velocity nodal locations
grid_vx = xv, expand_range(yc)
grid_vy = expand_range(xc), yv
grid_vxi = grid_vx, grid_vy

# Velocity defined on the grid
Vx = TA(backend)([-vi_stream(y) for x in grid_vx[1], y in grid_vx[2]])
Vy = TA(backend)([ vi_stream(x) for x in grid_vy[1], y in grid_vy[2]])
V = Vx, Vy

# Initialize marker chain
nxcell, min_xcell, max_xcell = 12, 6, 24
initial_elevation = Ly / 2
chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, initial_elevation)
method = RungeKutta2()

# time stepping
dt = 200.0
for _ in 1:25
    advect_markerchain!(chain, method, V, grid_vxi, dt)
end

# plot the chain
f = Figure()
ax = Axis(f[1, 1])
poly!(ax, Rect(0, 0, 1, 1), color = :lightgray)
px = chain.coords[1].data[:]
py = chain.coords[2].data[:]
scatter!(px, py, color = :black)
display(f)
```

The `scripts/rotating_marker_chain*.jl` files in the repository provide runnable Lagrangian
and semi-Lagrangian variants of this example.

## API

```@docs
MarkerChain
init_markerchain
fill_chain_from_chain!
fill_chain_from_vertices!
advect_markerchain!
semilagrangian_advection_markerchain!
resample!
compute_topography_vertex!
compute_rock_fraction!
interpolate_velocity_to_markerchain!
```
