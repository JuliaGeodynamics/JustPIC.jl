using MAT
# using GLMakie
using JLD2

using CUDA
using JustPIC
using JustPIC._3D
const backend = CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
# const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1-dx
    xF = x2+dx
    range(xI, xF, length=n+2)
end

function main(np)
    A   = matread("Taras/CornerFlow3D.mat")
    Vx = A["Vx"] |> CuArray
    Vy = A["Vy"] |> CuArray
    Vz = A["Vz"] |> CuArray
    V  = Vx, Vy, Vz

    lx   = 100 # Horizontal model size, m
    ly   = 100 # Vertical model size, m
    lz   = 100 # Vertical model size, m
    nx   = 40  # Horizontal grid resolution
    ny   = 40  # Vertical grid resolution
    nz   = 40  # Vertical grid resolution
    nx_v = nx+1
    ny_v = ny+1
    nz_v = nz+1
    dx   = lx/(nx) # Horizontal grid step, m
    dy   = ly/(ny) # Vertical grid step, m
    dz   = lz/(nz) # Vertical grid step, m
    x    = range(0, lx, length = nx_v)  # Horizontal coordinates of basic grid points, m
    y    = range(0, ly, length = ny_v)  # Vertical coordinates of basic grid points, m
    z    = range(0, lz, length = nz_v)  # Vertical coordinates of basic grid points, m

    # nodal centers
    xc, yc, zc = range(0+dx/2, lx-dx/2, length=nx), range(0+dy/2, ly-dy/2, length=ny), range(0+dz/2, lz-dz/2, length=nz)
    # staggered grid velocity nodal locations
    grid_vx = x, expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), y, expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), z
    # nodal vertices
    xvi = x, y, z

    grid_vxi = (
        grid_vx,
        grid_vy,
        grid_vz,
    )

    nxcell, max_xcell, min_xcell = np, 100, 1

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
    
    dt = min(dx / maximum(abs.(Array(Vx))),  dy / maximum(abs.(Array(Vy))),  dz / maximum(abs.(Array(Vz))));
    dt *= 0.75

    # ntime = 1000000
    ntime   = 10000#0
    np_Bi      = Int64[]
    np_MQS     = Int64[]
    np_LinP    = Int64[]
    empty_Bi   = Int64[]
    empty_MQS  = Int64[]
    empty_LinP = Int64[]
    full_Bi    = Int64[]
    full_MQS   = Int64[]
    full_LinP  = Int64[]
    
    time_Bi    = Float64[]
    time_MQS   = Float64[]
    time_LinP  = Float64[]
    
    for it in 1:ntime
        time_Bi0   = @elapsed advection!(particles1, RungeKutta2(), V, grid_vxi, dt)
        time_MQS0  = @elapsed advection_LinP!(particles2, RungeKutta2(), V, grid_vxi, dt)
        time_LinP0 = @elapsed advection_MQS!(particles3, RungeKutta2(), V, grid_vxi, dt)

        time_Bi0   += @elapsed move_particles!(particles1, xvi, ())
        time_MQS0  += @elapsed move_particles!(particles2, xvi, ())
        time_LinP0 += @elapsed move_particles!(particles3, xvi, ())

        
        # for (p, timer0) in zip( (particles1,particles2,particles3), (time_Bi0, time_MQS0, time_LinP0))
        #     timer0 += @elapsed move_particles!(p, xvi, ())
        #     push!(timer, timer0)
        #     # inject_particles!(p, (), xvi)
        # end
        # # inject && inject_particles!(particles, (), xvi)

        # total particles
        push!(np_Bi  , sum(particles1.index.data))
        push!(np_LinP, sum(particles2.index.data))
        push!(np_MQS , sum(particles3.index.data))

        push!(time_Bi,   time_Bi0)
        push!(time_MQS,  time_MQS0)
        push!(time_LinP, time_LinP0)

        # empty
        push!(empty_Bi  , sum([all(iszero,p) for p in Array(particles1.index)]))
        push!(empty_LinP, sum([all(iszero,p) for p in Array(particles2.index)]))
        push!(empty_MQS , sum([all(iszero,p) for p in Array(particles3.index)]))

        # full
        push!(full_Bi  , sum([sum(isone, p) > np for p in Array(particles1.index)]))
        push!(full_LinP, sum([sum(isone, p) > np for p in Array(particles2.index)]))
        push!(full_MQS , sum([sum(isone, p) > np for p in Array(particles3.index)]))
        
        if rem(it, 100) == 0
            @show it
        end
    end
    stats_Bi   = (; np = np_Bi  , empty = empty_Bi  , time = time_Bi,   full = full_Bi)
    stats_LinP = (; np = np_LinP, empty = empty_LinP, time = time_MQS,  full = full_LinP)
    stats_MQS  = (; np = np_MQS , empty = empty_MQS , time = time_LinP, full = full_MQS)

    return particles1, particles2, particles3, stats_Bi, stats_LinP, stats_MQS
end

function main0()
    np = 8
    for np in (4,8,12,16,20,24)

        println("Sarting with np = $np...")
        particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = main(np)

        jldsave(
            "Taras/CornerFlow3D_$(np)particles.jld2",
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
main0()