using MAT
# using GLMakie
using JLD2

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

function main(np, integrator)
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

    nxcell, max_xcell, min_xcell = np, 100, 1
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
    dt *= 0.75

    # ntime = 1000000
    ntime   = 100000
    np_Bi      = Int64[]
    np_MQS     = Int64[]
    np_LinP    = Int64[]
    empty_Bi   = Int64[]
    empty_MQS  = Int64[]
    empty_LinP = Int64[]
    full_Bi    = Int64[]
    full_MQS   = Int64[]
    full_LinP  = Int64[]
    
    injected_Bi    = Float64[]
    injected_MQS   = Float64[]
    injected_LinP  = Float64[]

    time_Bi    = Float64[]
    time_MQS   = Float64[]
    time_LinP  = Float64[]
    
    for _ in 1:ntime
        time_Bi0   = @elapsed advection!(particles1, integrator, V, grid_vxi, dt)
        time_MQS0  = @elapsed advection_LinP!(particles2, integrator, V, grid_vxi, dt)
        time_LinP0 = @elapsed advection_MQS!(particles3, integrator, V, grid_vxi, dt)

        time_Bi0   += @elapsed move_particles!(particles1, xvi, ())
        time_MQS0  += @elapsed move_particles!(particles2, xvi, ())
        time_LinP0 += @elapsed move_particles!(particles3, xvi, ())
        push!(time_Bi,   time_Bi0)
        push!(time_MQS,  time_MQS0)
        push!(time_LinP, time_LinP0)

        np1 = sum(particles1.index.data)
        np2 = sum(particles2.index.data)
        np3 = sum(particles3.index.data)

        for p in (particles1,particles2,particles3)
            inject_particles!(p, (), xvi)
        end

        np1 = sum(particles1.index.data) - np1
        np2 = sum(particles2.index.data) - np2
        np3 = sum(particles3.index.data) - np3
        
        # injected particles
        push!(injected_Bi  , np1)
        push!(injected_LinP, np2)
        push!(injected_MQS , np3)

        # total particles
        push!(np_Bi  , sum(particles1.index.data))
        push!(np_LinP, sum(particles2.index.data))
        push!(np_MQS , sum(particles3.index.data))

        # empty
        push!(empty_Bi  , sum([all(iszero,p) for p in particles1.index]))
        push!(empty_LinP, sum([all(iszero,p) for p in particles2.index]))
        push!(empty_MQS , sum([all(iszero,p) for p in particles3.index]))

        # full
        push!(full_Bi  , sum([sum(isone, p) > np for p in particles1.index]))
        push!(full_LinP, sum([sum(isone, p) > np for p in particles2.index]))
        push!(full_MQS , sum([sum(isone, p) > np for p in particles3.index]))
    end
    stats_Bi   = (; np = np_Bi  , empty = empty_Bi  , time = time_Bi,   injected = injected_Bi, full = full_Bi)
    stats_LinP = (; np = np_LinP, empty = empty_LinP, time = time_MQS,  injected = injected_LinP, full = full_LinP)
    stats_MQS  = (; np = np_MQS , empty = empty_MQS , time = time_LinP, injected = injected_MQS, full = full_MQS)

    return particles1, particles2, particles3, stats_Bi, stats_LinP, stats_MQS
end

function runner()
    for integrator in (RungeKutta2(), RungeKutta4()), np in (4,8,12,16,20,24)

        println("Sarting with np = $np...")
        particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = main(np, integrator)

        jldsave(
            "Taras/CornerFlow2D_$(np)particles.jld2",
            particles1 = particles1,
            particles2 = particles2,
            particles3 = particles3,
            stats_Lin = stats_Lin,
            stats_LinP = stats_LinP,
            stats_MQS = stats_MQS
        )
        println("...done with np = $np")

    end
end

runner()