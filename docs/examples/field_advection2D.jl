# # Field advection in 2D

using JustPIC

# Backend: `JustPIC.CPU`, or `CUDA.CUDABackend`, `AMDGPU.ROCBackend`, or
# `Metal.MetalBackend` after loading the corresponding package.
const backend = JustPIC.CPU

# Analytical flow used to advect the particles
vx_stream(x, y) = 250 * sin(π * x) * cos(π * y)
vy_stream(x, y) = -250 * cos(π * x) * sin(π * y)

# Model domain
n = 256           # number of nodes
nx = ny = n - 1   # number of cells in x and y
Lx = Ly = 1.0     # domain size
xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n) # cell vertices
dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]           # cell size
xci = xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1) # cell centers

# Staggered velocity grids: each staggered direction carries one ghost cell on
# either side, which `expand_range` adds to a 1D coordinate range.
function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1 - dx; sigdigits = 5)
    xF = round(x2 + dx; sigdigits = 5)
    return LinRange(xI, xF, n + 2)
end

grid_vx = xv, expand_range(yc) # staggered grid for Vx
grid_vy = expand_range(xc), yv # staggered grid for Vy

# Particles
nxcell = 24    # initial number of particles per cell
max_xcell = 48 # maximum number of particles per cell
min_xcell = 14 # minimum number of particles per cell
particles = init_particles(
    backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy,
)

# Velocity on the staggered grids and the advected field on the ghosted vertices
Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
xvi_particles = Array.(particles.xvi)
T = TA(backend)([y for x in xvi_particles[1], y in xvi_particles[2]])
V = Vx, Vy

# Particle field, initialized from the grid
particle_args = pT, = init_cell_arrays(particles, Val(1))
grid2particle!(pT, T, particles)

# Time loop
dt = min(dx / maximum(abs.(Array(Vx))), dy / maximum(abs.(Array(Vy)))) / 2
niter = 250
for it in 1:niter
    advection!(particles, RungeKutta2(), V, dt)  # advect particles
    move_particles!(particles, particle_args)    # move particles in the memory
    inject_particles!(particles, (pT,))          # inject particles if needed
    particle2grid!(T, pT, particles)             # interpolate particles to the grid
end
