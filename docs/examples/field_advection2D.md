# Field advection in 2D

First we load JustPIC

````julia
using JustPIC
````

and the correspondent 2D module (we could also use 3D by loading `JustPIC._3D`)

````julia
using JustPIC._2D
````

We need to specify what backend are we running our simulation on. For convenience we define the backend as a constant. In this case we use the CPU backend, but we could also use the CUDA (CUDABackend) or AMDGPU (AMDGPUBackend) backends.

````julia
const backend = CPUBackend
````

we define an analytical flow solution to advected our particles

````julia
vx_stream(x, y) =  250 * sin(π*x) * cos(π*y)
vy_stream(x, y) = -250 * cos(π*x) * sin(π*y)
````

define the model domain

````julia
n  = 256        # number of nodes
nx = ny = n-1   # number of cells in x and y
Lx = Ly = 1.0   # domain size
xvi = xv, yv = range(0, Lx, length=n), range(0, Ly, length=n) # cell vertices
xci = xc, yc = range(0+dx/2, Lx-dx/2, length=n-1), range(0+dy/2, Ly-dy/2, length=n-1)  # cell center
dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1] # cell size
````

JustPIC uses staggered grids for the velocity field, so we need to define the staggered grid for Vx and Vy. We

````julia
grid_vx = xv, expand_range(yc) # staggered grid for Vx
grid_vy = expand_range(xc), yv # staggered grid for Vy
````

where `expand_range` is a helper function that extends the range of a 1D array by one cell size in each direction

````julia
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

````julia
nxcell    = 24 # initial number of particles per cell
max_xcell = 48 # maximum number of particles per cell
min_xcell = 14 # minimum number of particles per cell
particles = init_particles(
    backend, nxcell, max_xcell, min_xcell, xvi..., dxi..., nx, ny
)
````

and the velocity and field we want to advect (on the staggered grid)

````julia
Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]]);
Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]]);
T  = TA(backend)([y for x in xv, y in yv]); # defined at the cell vertices
V  = Vx, Vy;
````

where `TA(backend)` will move the data to the specified backend (CPU, CUDA, or AMDGPU)

We also need to initialize the field `T` on the particles

````julia
particle_args = pT, = init_cell_arrays(particles, Val(1));
````

and we can use the function `grid2particle!` to interpolate the field `T` to the particles

````julia
grid2particle!(pT, xvi, T, particles);
````

we can now start the simulation

````julia
dt = min(dx / maximum(abs.(Array(Vx))),  dy / maximum(abs.(Array(Vy))));
niter = 250
for it in 1:niter
    advection_RK!(particles, V, grid_vx, grid_vy, dt, 2 / 3) # advect particles (α = 2 / 3)
    move_particles!(particles, xvi, particle_args) # move particles in the memory
    inject = check_injection(particles) # check if we need to inject particles
    inject && inject_particles!(particles, (pT, ), (T,), xvi) # inject particles if needed
    particle2grid!(T, pT, xvi, particles) # interpolate particles to the grid
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

