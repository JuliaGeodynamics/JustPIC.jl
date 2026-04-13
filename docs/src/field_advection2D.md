# Field advection in 2D

First, load JustPIC:

```julia
using JustPIC
```

Then load the 2D module. For 3D models, use `JustPIC._3D` instead.

```julia
using JustPIC._2D
```

Choose the backend used for particle storage and kernels. This example uses the
CPU backend, but CUDA and AMDGPU backends can be used when the corresponding
extension packages are loaded.

```julia
const backend = JustPIC.CPUBackend
```

Define an analytical flow used to advect the particles:

```julia
vx_stream(x, y) =  250 * sin(π*x) * cos(π*y)
vy_stream(x, y) = -250 * cos(π*x) * sin(π*y)
```

Define the model domain:

```julia
n  = 256        # number of nodes
nx = ny = n-1   # number of cells in x and y
Lx = Ly = 1.0   # domain size
xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n) # cell vertices
dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1] # cell size
xci = xc, yc = LinRange(0+dx/2, Lx-dx/2, n-1), LinRange(0+dy/2, Ly-dy/2, n-1) # cell centers
```

JustPIC uses staggered velocity grids, so we define one coordinate tuple for
each velocity component:

```julia
grid_vx = xv, expand_range(yc) # staggered grid for Vx
grid_vy = expand_range(xc), yv # staggered grid for Vy
```

Here `expand_range` extends a 1D coordinate range by one cell size on both sides:

```julia
function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    LinRange(xI, xF, n + 2)
end
```

Next, initialize the particles:

```julia
nxcell    = 24 # initial number of particles per cell
max_xcell = 48 # maximum number of particles per cell
min_xcell = 14 # minimum number of particles per cell
particles = init_particles(
    backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy
)
```

Define the velocity field on the staggered grids and the scalar field on
vertices:

```julia
Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]]);
Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]]);
T  = TA(backend)([y for x in xv, y in yv]); # defined at the cell vertices
V  = Vx, Vy;
nothing #hide
```

`TA(backend)` converts the data to the array type associated with the selected
backend.

We also need to initialize the field `T` on the particles

```julia
particle_args = pT, = init_cell_arrays(particles, Val(1));
nothing #hide
```

Use `grid2particle!` to interpolate `T` to the particles:

```julia
grid2particle!(pT, T, particles);
nothing #hide
```

We can now start the time loop:

```julia
dt = min(dx / maximum(abs.(Array(Vx))),  dy / maximum(abs.(Array(Vy)))) / 2
niter = 250
for it in 1:niter
    advection!(particles, RungeKutta2(), V, dt)                     # advect particles
    move_particles!(particles, particle_args)                       # move particles in the memory
    inject_particles!(particles, (pT, ))                            # inject particles if needed
    particle2grid!(T, pT, particles)                                # interpolate particles to the grid
end
```

# Pure shear in 2D

An example of two-dimensional pure shear flow is provided in
[`scripts/pureshear_ALE.jl`](https://github.com/JuliaGeodynamics/JustPIC.jl/blob/main/scripts/pureshear_ALE.jl).
The velocity field is set to:

$v_{x} = \dot{\varepsilon} x$

$v_{y} = -\dot{\varepsilon} y$

where $\dot{\varepsilon}$ is the pure shear strain rate applied at the boundaries. A positive value of $\dot{\varepsilon}$ leads to horizontal extension, while negative values correspond to horizontal compression.

The `ALE` switch (Arbitrary Lagrangian-Eulerian) controls model-box deformation.
If `ALE=false`, the model dimensions remain constant over time. If `ALE=true`,
the model domain deforms with the background pure-shear rate.
  
