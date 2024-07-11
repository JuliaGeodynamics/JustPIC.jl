using DelimitedFiles
using GLMakie

using JustPIC
using JustPIC._2D
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# Threads is the default backend, 
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"), 
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend


function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    range(xI, xF, length=n+2)
end

# read velocity from Taras
Vx = Array(readdlm("Taras/Vx_v.txt", '\t')')
Vy = Array(readdlm("Taras/Vy_v.txt", '\t')')

# heatmap(Vx[:, 1:end-1])
# heatmap(Vx)

Vx = Vx[1:end-1, :]
Vy = Vy[:, 1:end-1]
V  = Vx, Vy

lx   = 100 # Horizontal model size, m
ly   = 100 # Vertical model size, m
nx   = 40  # Horizontal grid resolution
ny   = 40  # Vertical grid resolution
nx_v = nx+1
ny_v = ny+1
dx   = lx/(nx) # Horizontal grid step, m
dy   = ly/(ny) # Vertical grid step, m
x    = range( 0, lx, length = nx_v)  # Horizontal coordinates of basic grid points, m
y    = range( 0, ly, length = ny_v)  # Vertical coordinates of basic grid points, m
# xvx  = range( 0, lx, length = nx_v)  # Horizontal coordinates of vx grid points, m
# yvx  = range( -dy/2, ly+dy/2, length = ny_v+1) # Vertical coordinates of vx grid points, m
# xvy  = range( -dx/2, lx+dx/2, length = nx_v+1) # Horizontal coordinates of vy grid points, m
# yvy  = range( 0, ly, length = ny_v) # Vertical coordinates of vy grid points, m
# xp   = range( -dx/2, lx+dx/2, length = nx + 1) # Horizontal coordinates of P grid points, m
# yp   = range( -dy/2, ly+dy/2, length = ny + 1) # Vertical coordinates of P grid points, m

# nodal centers
xc, yc = range(0+dx/2, lx-dx/2, length=nx), range(0+dy/2, ly-dy/2, length=ny)
# staggered grid velocity nodal locations
grid_vx = x, expand_range(yc)
grid_vy = expand_range(xc), y

xvi = x, y

grid_vxi = (
    grid_vx,
    grid_vy,
)

nxcell, max_xcell, min_xcell = 8, 30, 2
# nodal vertices
xvi = x, y 
dxi = dx, dy

T  = TA(backend)([i > j ? 2e0 : 1e0  for i in 1:nx_v, j in 1:ny_v]);
dt = min(dx / maximum(abs.(Array(Vx))),  dy / maximum(abs.(Array(Vy)))) / 2

particles = init_particles(
    backend, nxcell, max_xcell, min_xcell, xvi...,
)
particle_args = pT, = init_cell_arrays(particles, Val(1));

grid2particle!(pT, xvi, T, particles);
pxx, pyy = particles.coords
scatter(pxx.data[:], pyy.data[:], color = pT.data[:], markersize = 4)

niter = 1
m = 3
# ntime = 1000000
ntime = 10000
for it in 1:ntime
    if m == 1
        advection!(particles, RungeKutta2(), V, grid_vxi, dt)
    elseif m == 2
        advection_LinP!(particles, RungeKutta2(), V, grid_vxi, dt)
    elseif m == 3
        advection_MQS!(particles, RungeKutta2(), V, grid_vxi, dt)
    end
    move_particles!(particles, xvi, particle_args)
    # inject_particles!(particles, (pT, ), xvi)
end

# @show (m, sum(particles.index.data))
pxx, pyy = particles.coords
scatter(pxx.data[:], pyy.data[:], markersize = 4)

