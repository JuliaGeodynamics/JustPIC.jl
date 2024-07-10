using DelimitedFiles
using GLMakie

using JustPIC
using JustPIC._2D
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# Threads is the default backend, 
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"), 
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

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
nx   = 41  # Horizontal grid resolution
ny   = 41  # Vertical grid resolution
nx_v = nx+1
ny_v = ny+1
dx   = lx/(nx-1) # Horizontal grid step, m
dy   = ly/(ny-1) # Vertical grid step, m
x    = range( 0, lx, length = nx_v)  # Horizontal coordinates of basic grid points, m
y    = range( 0, ly, length = ny_v)  # Vertical coordinates of basic grid points, m
xvx  = range( 0, lx, length = nx_v)  # Horizontal coordinates of vx grid points, m
yvx  = range( -dy/2, ly+dy/2, length = ny_v+1) # Vertical coordinates of vx grid points, m
xvy  = range( -dx/2, lx+dx/2, length = nx_v+1) # Horizontal coordinates of vy grid points, m
yvy  = range( 0, ly, length = ny_v) # Vertical coordinates of vy grid points, m
xp   = range( -dx/2, lx+dx/2, length = nx + 1) # Horizontal coordinates of P grid points, m
yp   = range( -dy/2, ly+dy/2, length = ny + 1) # Vertical coordinates of P grid points, m

xvi = x, y

grid_vxi = (
    (xvx, yvx),
    (xvy , yvy),
)

nxcell, max_xcell, min_xcell = 4, 30, 0
# nodal vertices
xvi = x, y 
dxi = dx, dy
# staggered grid velocity nodal locations
grid_vx = grid_vxi[1]
grid_vy = grid_vxi[2]


T  = TA(backend)([i > j ? 2e0 : 1e0  for i in 1:nx_v, j in 1:ny_v]);
dt = min(dx / maximum(abs.(Array(Vx))),  dy / maximum(abs.(Array(Vy)))) / 2

particles = init_particles(
    backend, nxcell, max_xcell, min_xcell, xvi...,
)
particle_args = pT, = init_cell_arrays(particles, Val(1));

grid2particle!(pT, xvi, T, particles);
pxx, pyy = particles.coords
scatter(pxx.data[:], pyy.data[:], color = pT.data[:], markersize = 4)

niter = 1000
m = 3
for it in 1:niter
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

pxx, pyy = particles.coords
scatter(pxx.data[:], pyy.data[:], markersize = 4)

###
# p = 2
# x = 0e0:3e0
# F = hcat(x.^p, x.^2)

# v = F[2,1], F[3,1], F[2,2], F[3,2] 
# t = rand(), rand()

# dx = dy = 1
# px = py = 1.25
# t  = ((px-1)/dx, (py-1)/dy)

# MQS(F, v, t, 2, 1, Val(1))
# lerp(v, t)

# px^2

# for _ in 1:1000
#     t = rand(), rand()
#     check = MQS(F, v, t, 2, 1, Val(1)) ≈ lerp(v, t)
#     if !check
#         @show t
#     end
# end

# t = (0.3534920817052422, 0.08331497681241812)
# MQS(F, v, t, 2, 1, Val(1))
# lerp(v, t)


# ## case x-1
# px, py = 3.3263, 0.7350
# dx, dy = 2.5, 2.5
# x, y   = 2.5, -1.25
# t = ((px-x)/dx, (py-y)/dy)

# Fbot = [0; 0.1566;    0.2215;    0.1394].*1e-9
# Ftop = [0; 0.1566;    0.2215;    0.1394].*1e-9
# F = hcat(Fbot, Ftop)
# v = F[2,1], F[3,1], F[2,2], F[3,2] 

# MQS(F, v, t, 2, 1, Val(1))


# ## case x-2
# px, py = 1.8333, 0.4875
# dx, dy = 2.5, 2.5
# x, y   = 0, -1.25
# t = ((px-x)/dx, (py-y)/dy)

# Fbot = [0; 0;    0.1566;    0.2215].*1e-9
# Ftop = [0; 0;    0.1566;    0.2215].*1e-9
# F = hcat(Fbot, Ftop)
# v = F[2,1], F[3,1], F[2,2], F[3,2] 

# MQS(F, v, t, 2, 1, Val(1))

# ## case y-1
# px, py = 1.5030,  3.5101
# dx, dy = 2.5, 2.5
# x, y   = 1.25, 2.5
# t = ((px-x)/dx, (py-y)/dy)

# # Fleft  = [0  0.1566   -0.3873   -0.6990].*1e-9
# # Fright = [0 -0.0649   -0.2192   -0.5668].*1e-9
# F = [
#     0 -0.1566   -0.3873   -0.6990
#     0 -0.0649   -0.2192   -0.5668
# ].*1e-9
# v = F[1,2], F[2,2], F[1,3], F[2,3]

# MQS(F, v, t, 1, 2, Val(2))


# v_left = (1.566e-10, -3.873e-10)
# v_right = (-6.49e-11, -2.1920000000000003e-10)

# lerp(v_left, (0.404,))