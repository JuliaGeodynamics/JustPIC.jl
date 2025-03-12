using MAT
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

function main()
    A   = matread("Taras/CornerFlow2D.mat")
    Vx = A["Vx"]
    Vy = A["Vy"]
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

    nxcell, max_xcell, min_xcell = 4, 20, 1
    # nodal vertices
    xvi = x, y 

    # dt = 2.2477e7
    particles1 = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )
    particles2 = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )
    particles3 = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )
    
    dt = min(dx / maximum(abs.(Array(Vx))),  dy / maximum(abs.(Array(Vy))));
    # dt *= 0.75

    # ntime = 1000000
    ntime   = 10000
    np_Bi      = Int64[]
    np_MQS     = Int64[]
    np_LinP    = Int64[]
    empty_Bi   = Int64[]
    empty_MQS  = Int64[]
    empty_LinP = Int64[]
    full_Bi    = Int64[]
    full_MQS   = Int64[]
    full_LinP  = Int64[]
    
    for it in 1:ntime
        # if m == 1
            advection!(particles1, RungeKutta2(), V, grid_vxi, dt)
        # elseif m == 2
            advection_LinP!(particles2, RungeKutta2(), V, grid_vxi, dt)
        # elseif m == 3
            advection_MQS!(particles3, RungeKutta2(), V, grid_vxi, dt)
        # end
        for p in (particles1,particles2,particles3)
            move_particles!(p, xvi, ())
            # inject_particles!(p, (), xvi)
        end
        # inject && inject_particles!(particles, (), xvi)

        # total particles
        push!(np_Bi  , sum(particles1.index.data))
        push!(np_MQS , sum(particles2.index.data))
        push!(np_LinP, sum(particles3.index.data))

        # empty
        push!(empty_Bi  , sum([all(iszero,p) for p in particles1.index]))
        push!(empty_MQS , sum([all(iszero,p) for p in particles2.index]))
        push!(empty_LinP, sum([all(iszero,p) for p in particles3.index]))

        # full
        push!(full_Bi  , sum([sum(isone, p) for p in particles1.index]))
        push!(full_MQS , sum([sum(isone, p) for p in particles2.index]))
        push!(full_LinP, sum([sum(isone, p) for p in particles3.index]))
    end
    stats_Bi   = (; np = np_Bi  , empty = empty_Bi  , full = full_Bi)
    stats_LinP = (; np = np_LinP, empty = empty_LinP, full = full_LinP)
    stats_MQS  = (; np = np_MQS , empty = empty_MQS , full = full_MQS)

    return particles1, particles2, particles3, stats_Bi, stats_LinP, stats_MQS
end

particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = main()

d1 = [count(p) for p in particles1.index];
d2 = [count(p) for p in particles2.index];
d3 = [count(p) for p in particles3.index];

heatmap(d1 ./ 8, colorrange = (0, 2))
heatmap(d2 ./ 8, colorrange = (0, 2))
heatmap(d3 ./ 8, colorrange = (0, 2))

scatterlines( stats_Lin.np , markersize = 4)
scatterlines!(stats_LinP.np, markersize = 4)
scatterlines!(stats_MQS.np , markersize = 4)

scatterlines( stats_Lin.empty  ./ 40^2, markersize = 4)
scatterlines!(stats_LinP.empty ./ 40^2, markersize = 4)
scatterlines!(stats_MQS.empty  ./ 40^2, markersize = 4)

scatterlines( stats_Lin.full  ./ 40^2, markersize = 4)
scatterlines!(stats_LinP.full ./ 40^2, markersize = 4)
scatterlines!(stats_MQS.full  ./ 40^2, markersize = 4)

pxx, pyy  = particles1.coords
scatter( pxx.data[:], pyy.data[:], markersize = 4)

pxx, pyy  = particles2.coords
scatter( pxx.data[:], pyy.data[:], markersize = 4)

pxx, pyy  = particles3.coords
scatter( pxx.data[:], pyy.data[:], markersize = 4)