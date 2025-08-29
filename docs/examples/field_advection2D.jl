# # Field advection in 2D 

# First we load JustPIC
using JustPIC

# and the correspondent 2D module (we could also use 3D by loading `JustPIC._3D`)
using JustPIC._2D

# We need to specify what backend are we running our simulation on. For convenience we define the backend as a constant. In this case we use the CPU backend, but we could also use the CUDA (CUDABackend) or AMDGPU (AMDGPUBackend) backends.
const backend = JustPIC.CPUBackend 

# we define an analytical flow solution to advected our particles
vx_stream(x, y) =  250 * sin(π*x) * cos(π*y)
vy_stream(x, y) = -250 * cos(π*x) * sin(π*y)

# define the model domain
n   = 256        # number of nodes
nx  = ny = n-1   # number of cells in x and y
ni  = nx, ny
Lx  = Ly = 1.0   # domain size
Li  = Lx, Ly
dxi = dx, dy = Li ./ ni
xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n) # cell vertices
xci = xc, yc = LinRange(0+dx/2, Lx-dx/2, n-1), LinRange(0+dy/2, Ly-dy/2, n-1) # cell centers

# JustPIC uses staggered grids for the velocity field, so we need to define the staggered grid for Vx and Vy. We 
grid_vx = xv, expand_range(yc) # staggered grid for Vx
grid_vy = expand_range(xc), yv # staggered grid for Vy

# where `expand_range` is a helper function that extends the range of a 1D array by one cell size in each direction
function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    LinRange(xI, xF, n + 2)
end

# Next we initialize the particles
nxcell    = 24 # initial number of particles per cell
max_xcell = 48 # maximum number of particles per cell
min_xcell = 14 # minimum number of particles per cell
particles = init_particles(
    backend, nxcell, max_xcell, min_xcell, xvi..., 
)

# and the velocity and field we want to advect (on the staggered grid)
Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]]);
Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]]);
T  = TA(backend)([y for x in xv, y in yv]); # defined at the cell vertices
V  = Vx, Vy;
# where `TA(backend)` will move the data to the specified backend (CPU, CUDA, or AMDGPU)

# We also need to initialize the field `T` on the particles
particle_args = pT, = init_cell_arrays(particles, Val(1));
# and we can use the function `grid2particle!` to interpolate the field `T` to the particles
grid2particle!(pT, xvi, T, particles);
    
# we can now start the simulation
dt = min(dx / maximum(abs.(Array(Vx))),  dy / maximum(abs.(Array(Vy)))) * 0.5;
niter = 250
for it in 1:niter
    advection!(particles, RungeKutta2(), V, (grid_vx, grid_vy), dt) # advect particles
    move_particles!(particles, xvi, particle_args)                  # move particles in the memory
    inject_particles!(particles, (pT, ), xvi)                       # inject particles if needed
    particle2grid!(T, pT, xvi, particles)                           # interpolate particles to the grid
end
