# Field advection in 3D

First, load JustPIC:

```julia
using JustPIC
```

Then load the 3D module:

```julia
using JustPIC._3D
```

Choose the backend used for particle storage and kernels. This example uses the
CPU backend, but CUDA and AMDGPU backends can be used when the corresponding
extension packages are loaded.

```julia
const backend = JustPIC.CPUBackend
```

Define an analytical flow used to advect the particles:

```julia
vx_stream(x, z) =  250 * sin(π*x) * cos(π*z)
vy_stream(x, z) =  0.0
vz_stream(x, z) = -250 * cos(π*x) * sin(π*z)
```

Define the model domain:

```julia
n  = 64             # number of nodes
nx  = ny = nz = n-1 # number of cells in x and y
Lx  = Ly = Lz = 1.0 # domain size
ni  = nx, ny, nz
Li  = Lx, Ly, Lz

xvi = xv, yv, zv = ntuple(i -> LinRange(0, Li[i], n), Val(3)) # cell vertices
dxi = dx, dy, dz = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3)) # cell size
xci = xc, yc, zc = ntuple(i -> LinRange(0+dxi[i]/2, Li[i]-dxi[i]/2, ni[i]), Val(3)) # cell centers
```

JustPIC uses staggered velocity grids, so we define one coordinate tuple for
each velocity component:

```julia
grid_vx = xv              , expand_range(yc), expand_range(zc) # staggered grid for Vx
grid_vy = expand_range(xc), yv              , expand_range(zc) # staggered grid for Vy
grid_vz = expand_range(xc), expand_range(yc), zv               # staggered grid for Vz
```

Here `expand_range` extends a 1D coordinate range by one cell size on both sides:

```julia
function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    LinRange(xI, xF, n+2)
end
```

Next, initialize the particles:

```julia
nxcell    = 24 # initial number of particles per cell
max_xcell = 48 # maximum number of particles per cell
min_xcell = 14 # minimum number of particles per cell
particles = init_particles(
    backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy, grid_vz
)
```

Define the velocity field on the staggered grids and the scalar field on
vertices:

```julia
Vx = TA(backend)([vx_stream(x, z) for x in grid_vx[1], y in grid_vx[2], z in grid_vx[3]])
Vy = TA(backend)([vy_stream(x, z) for x in grid_vy[1], y in grid_vy[2], z in grid_vy[3]])
Vz = TA(backend)([vz_stream(x, z) for x in grid_vz[1], y in grid_vz[2], z in grid_vz[3]])
T  = TA(backend)([z for x in xv, y in yv, z in zv]) # defined at the cell vertices
V  = Vx, Vy, Vz
```

`TA(backend)` converts the data to the array type associated with the selected
backend.

We also need to initialize the field `T` on the particles

```julia
particle_args = pT, = init_cell_arrays(particles, Val(1));
```

Use `grid2particle!` to interpolate `T` to the particles:

```julia
grid2particle!(pT, T, particles)
```

We can now start the time loop:

```julia
dt = min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)), dz / maximum(abs.(Vz))) / 2

niter = 250
for it in 1:niter
    advection!(particles, RungeKutta2(), V, dt)                               # advect particles
    move_particles!(particles, particle_args)                                # move particles in the memory
    inject_particles!(particles, (pT, ))                                     # inject particles if needed
    particle2grid!(T, pT, particles)                                         # interpolate particles to the grid
end
```
