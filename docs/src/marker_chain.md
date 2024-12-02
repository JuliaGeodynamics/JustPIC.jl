# Marker chain

In, e.g., geodynamic modeling is often useful to track interfaces between different materials, such as the topographic profile. We can use the `MarkerChain` object in JustPIC.jl to define and advect surfaces (not closed-polygons) in two-dimensional models. 

We can instantiate a chain object with a given constant elevation `h` as:

`chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, h)`

where `backend` is the device backend, `nxcell` is the initial number of makers per cell, `min_xcell` and `max_xcell` are the minimum and maximum number of particles allowed per cell, respectively, and `xv` is the grid corresponding to the vertices of the the grid.

We can also fill an existing `MarkerChain` with a given topographic profile:
```julia
# create topographic profile
x      = LinRange(0, 1, 200)
topo_x = LinRange(0, 1, 200)
topo_y = @. sin(2π*topo_x) * 0.1

# fill the chain with the topographic profile` 
fill_chain!(chain, topo_x, topo_y)
```

Finally, the marker chain can be advected as follows:

```julia
advect_markerchain!(chain, method, velocity, grid_vxi, dt)
```

where `method` is the time integration method of the advection equation, `velocity` is a tuple containing the arrays of the velocity field, `grid_vxi` is a tuple containing the grids of the velocity components on the staggered grid, and `t` is the time step.

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