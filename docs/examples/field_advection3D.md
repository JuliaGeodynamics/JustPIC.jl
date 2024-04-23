```@meta
EditURL = "docs/examples/field_advection3D.jl"
```

# Field advection in 3D

First we load JustPIC

````@example field_advection3D
using JustPIC
````

and the correspondent 3D module

````@example field_advection3D
using JustPIC._3D
````

We need to specify what backend are we running our simulation on. For convenience we define the backend as a constant. In this case we use the CPU backend, but we could also use the CUDA (CUDABackend) or AMDGPU (AMDGPUBackend) backends.

````@example field_advection3D
const backend = CPUBackend
````

we define an analytical flow solution to advected our particles

````@example field_advection3D
vx_stream(x, z) =  250 * sin(π*x) * cos(π*z)
vy_stream(x, z) =  0.0
vz_stream(x, z) = -250 * cos(π*x) * sin(π*z)
````

define the model domain

````@example field_advection3D
n  = 64             # number of nodes
nx  = ny = nz = n-1 # number of cells in x and y
Lx  = Ly = Lz = 1.0 # domain size
ni  = nx, ny, nz
Li  = Lx, Ly, Lz

xvi = xv, yv, zv = ntuple(i -> range(0, Li[i], length=n), Val(3)) # cell vertices
xci = xc, yc, zc = ntuple(i -> range(0+dxi[i]/2, Li[i]-dxi[i]/2, length=ni[i]), Val(3)) # cell centers
dxi = dx, dy, dz = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3)) # cell size
````

JustPIC uses staggered grids for the velocity field, so we need to define the staggered grid for Vx and Vy. We

````@example field_advection3D
grid_vx = xv              , expand_range(yc), expand_range(zc) # staggered grid for Vx
grid_vy = expand_range(xc), yv              , expand_range(zc) # staggered grid for Vy
grid_vz = expand_range(xc), expand_range(yc), zv               # staggered grid for Vy
````

where `expand_range` is a helper function that extends the range of a 1D array by one cell size in each direction

````@example field_advection3D
function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    range(xI, xF, length=n+2)
end
````

Next we initialize the particles

````@example field_advection3D
nxcell    = 24 # initial number of particles per cell
max_xcell = 48 # maximum number of particles per cell
min_xcell = 14 # minimum number of particles per cell
particles = init_particles(
    backend, nxcell, max_xcell, min_xcell, xvi, dxi, ni
)
````

and the velocity and field we want to advect (on the staggered grid)

````@example field_advection3D
Vx = TA(backend)([vx_stream(x, z) for x in grid_vx[1], y in grid_vx[2], z in grid_vx[3]])
Vy = TA(backend)([vy_stream(x, z) for x in grid_vy[1], y in grid_vy[2], z in grid_vy[3]])
Vz = TA(backend)([vz_stream(x, z) for x in grid_vz[1], y in grid_vz[2], z in grid_vz[3]])
T  = TA(backend)([z for x in xv, y in yv, z in zv]) # defined at the cell vertices
V  = Vx, Vy, Vz
````

where `TA(backend)` will move the data to the specified backend (CPU, CUDA, or AMDGPU)

We also need to initialize the field `T` on the particles

````@example field_advection3D
particle_args = pT, = init_cell_arrays(particles, Val(1));
nothing #hide
````

and we can use the function `grid2particle!` to interpolate the field `T` to the particles

````@example field_advection3D
grid2particle!(pT, xvi, T, particles)
````

we can now start the simulation

````@example field_advection3D
dt = min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)), dz / maximum(abs.(Vz))) / 2

niter = 250
for it in 1:niter
    advection!(particles, RungeKutta2(), V, (grid_vx, grid_vy, grid_vz), dt) # advect particles
    move_particles!(particles, xvi, particle_args)                           # move particles in the memory
    inject_particles!(particles, (pT, ), xvi)                                # inject particles if needed
    particle2grid!(T, pT, xvi, particles)                                    # interpolate particles to the grid
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

