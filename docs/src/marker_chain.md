# Marker chain

`MarkerChain` is the 2D interface-tracking container in JustPIC. It is intended
for open surfaces such as topography or material boundaries, not for closed
polygons.

You can instantiate a chain object with a constant elevation `h` as:

`chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, h)`

Here `backend` is the device backend, `nxcell` is the initial number of markers
per cell, `min_xcell` and `max_xcell` are the occupancy bounds used by
resampling, and `xv` is the horizontal vertex grid.

You can also overwrite an existing `MarkerChain` with a sampled topographic
profile:
```julia
# create topographic profile
x      = LinRange(0, 1, 200)
topo_x = LinRange(0, 1, 200)
topo_y = @. sin(2π*topo_x) * 0.1

# fill the chain with the topographic profile` 
fill_chain_from_chain!(chain, topo_x, topo_y)
```

Once initialized, the high-level advection helper is:

```julia
advect_markerchain!(chain, method, velocity, grid_vxi, dt)
```

It advances markers, moves them back into the correct cells, resamples the
chain, rebuilds vertex topography, and reconstructs the marker representation.

## Example

```julia
using JustPIC
using JustPIC._2D
using GLMakie

const backend = JustPIC.CPUBackend

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    LinRange(xI, xF, n+2)
end

# velocity field
vi_stream(x) =  π*1e-5 * (x - 0.5)

# Initialize domain & grids
n        = 51
Lx       = Ly = 1.0
# nodal vertices
xvi      = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
dxi      = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
# nodal centers
xc, yc   = LinRange(0+dx/2, Lx-dx/2, n-1), LinRange(0+dy/2, Ly-dy/2, n-1)
# staggered grid velocity nodal locations
grid_vx  = xv, expand_range(yc)
grid_vy  = expand_range(xc), yv
grid_vxi = grid_vx, grid_vy

# Velocity defined on the grid
Vx       = TA(backend)([-vi_stream(y) for x in grid_vx[1], y in grid_vx[2]]);
Vy       = TA(backend)([ vi_stream(x) for x in grid_vy[1], y in grid_vy[2]]);
V        = Vx, Vy;

# Initialize marker chain
nxcell, min_xcell, max_xcell = 12, 6, 24
initial_elevation = Ly/2
chain             = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, initial_elevation);
method            = RungeKutta2()

# time stepping
dt       = 200.0

for _ in 1:25
    advect_markerchain!(chain, method, V, grid_vxi, dt)
end

# plotting the chain
f = Figure()
ax = Axis(f[1, 1])
poly!(
    ax,
    Rect(0, 0, 1, 1),
    color=:lightgray,
)
px = chain.coords[1].data[:];
py = chain.coords[2].data[:];
scatter!(px, py, color=:black)
display(f)
```
