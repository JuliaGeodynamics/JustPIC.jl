# # Field advection in 3D

# First we load JustPIC
using JustPIC

# and the correspondent 3D module
using JustPIC._3D

# We need to specify what backend are we running our simulation on. For convenience we define the backend as a constant. In this case we use the CPU backend, but we could also use the CUDA (CUDABackend) or AMDGPU (AMDGPUBackend) backends.
const backend = JustPIC.CPUBackend

# we define an analytical flow solution to advected our particles
vx_stream(x, z) = 250 * sin(π * x) * cos(π * z)
vy_stream(x, z) = 0.0
vz_stream(x, z) = -250 * cos(π * x) * sin(π * z)

# define the model domain
n = 64             # number of nodes
nx = ny = nz = n - 1 # number of cells in x and y
Lx = Ly = Lz = 1.0 # domain size
ni = nx, ny, nz
Li = Lx, Ly, Lz
dxi = dx, dy, dz = Li ./ ni # cell size

xvi = xv, yv, zv = ntuple(i -> LinRange(0, Li[i], n), Val(3)) # cell vertices
xci = xc, yc, zc = ntuple(i -> LinRange(0 + dxi[i] / 2, Li[i] - dxi[i] / 2, ni[i]), Val(3)) # cell centers

# JustPIC uses staggered grids for the velocity field, so we need to define the staggered grid for Vx and Vy. We
grid_vx = xv, expand_range(yc), expand_range(zc) # staggered grid for Vx
grid_vy = expand_range(xc), yv, expand_range(zc) # staggered grid for Vy
grid_vz = expand_range(xc), expand_range(yc), zv # staggered grid for Vy

# where `expand_range` is a helper function that extends the range of a 1D array by one cell size in each direction
function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1 - dx; sigdigits = 5)
    xF = round(x2 + dx; sigdigits = 5)
    return LinRange(xI, xF, n + 2)
end

# Next we initialize the particles
nxcell = 24 # initial number of particles per cell
max_xcell = 48 # maximum number of particles per cell
min_xcell = 14 # minimum number of particles per cell
particles = init_particles(
    backend, nxcell, max_xcell, min_xcell, xvi...
)

# and the velocity and field we want to advect (on the staggered grid)
Vx = TA(backend)([vx_stream(x, z) for x in grid_vx[1], y in grid_vx[2], z in grid_vx[3]])
Vy = TA(backend)([vy_stream(x, z) for x in grid_vy[1], y in grid_vy[2], z in grid_vy[3]])
Vz = TA(backend)([vz_stream(x, z) for x in grid_vz[1], y in grid_vz[2], z in grid_vz[3]])
T = TA(backend)([z for x in xv, y in yv, z in zv]) # defined at the cell vertices
V = Vx, Vy, Vz
# where `TA(backend)` will move the data to the specified backend (CPU, CUDA, or AMDGPU)

# We also need to initialize the field `T` on the particles
particle_args = pT, = init_cell_arrays(particles, Val(1));
# and we can use the function `grid2particle!` to interpolate the field `T` to the particles
grid2particle!(pT, xvi, T, particles)

# we can now start the simulation
dt = min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)), dz / maximum(abs.(Vz))) / 2

niter = 250
for it in 1:niter
    advection!(particles, RungeKutta2(), V, (grid_vx, grid_vy, grid_vz), dt) # advect particles
    move_particles!(particles, xvi, particle_args)                           # move particles in the memory
    inject_particles!(particles, (pT,), xvi)                                # inject particles if needed
    particle2grid!(T, pT, xvi, particles)                                    # interpolate particles to the grid
end
