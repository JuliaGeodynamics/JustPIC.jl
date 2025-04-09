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
    xI = x1-dx
    xF = x2+dx
    range(xI, xF, length=n+2)
end

# read velocity from Taras
Vx0 = Array(readdlm("Taras/Vx.txt", '\t')')
Vy0 = Array(readdlm("Taras/Vy.txt", '\t')')

Vx = @. (Vx0[1:end-1, :] + Vx0[2:end, :])/2
Vy = @. (Vy0[:, 1:end-1] + Vy0[:, 2:end])/2
V  = Vx, Vy

# function main(V, m, inject)
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

    nxcell, max_xcell, min_xcell = 4, 90, 1
    # nodal vertices
    xvi = x, y 

    dt = 2.2477e7
    particles1 = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )
    particles2 = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )
    particles3 = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )
    # advection!(particles, RungeKutta2(), V, grid_vxi, dt)

    # particle_args = pT, = init_cell_arrays(particles, Val(1));

    niter   = 1
    m       = 1
    inject  = true
    # ntime = 1000000
    ntime   = 100000
    @show inject
    np      = zeros(Int64, ntime)
    npMQS   = zeros(Int64, ntime)
    npLinP  = zeros(Int64, ntime)
    method  = RungeKutta4()
    for it in 1:ntime
        # if m == 1
            advection!(particles1, method, V, grid_vxi, dt)
        # elseif m == 2
            advection_LinP!(particles2, method, V, grid_vxi, dt)
        # elseif m == 3
            advection_MQS!(particles3, method, V, grid_vxi, dt)
        # end
        for p in (particles1,particles2,particles3)
            move_particles!(p, xvi, ())
            # inject_particles!(p, (), xvi)
        end
        # inject && inject_particles!(particles, (), xvi)

        np[it]     = sum(particles1.index.data)
        npMQS[it]  = sum(particles2.index.data)
        npLinP[it] = sum(particles3.index.data)
    end
#     return particles, np
# end

# d1 = [count(p) for p in particles1.index];
# d2 = [count(p) for p in particles2.index];
# d3 = [count(p) for p in particles3.index];

# heatmap(d1)
# heatmap(d2)
# heatmap(d3)
scatterlines(np, markersize = 4)
scatterlines!(npMQS, markersize = 4)
scatterlines!(npLinP, markersize = 4)

fig = Figure(size=(800,1400))
ax1 = Axis(fig[1,1], aspect = DataAspect())
ax2 = Axis(fig[2,1], aspect = DataAspect())
ax3 = Axis(fig[3,1], aspect = DataAspect())

pxx, pyy  = particles1.coords
scatter!(ax1, pxx.data[:], pyy.data[:], markersize = 4)

pxx, pyy  = particles2.coords
scatter!(ax2, pxx.data[:], pyy.data[:], markersize = 4)

pxx, pyy  = particles3.coords
scatter!(ax3, pxx.data[:], pyy.data[:], markersize = 4)

fig