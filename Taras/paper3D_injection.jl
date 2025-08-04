using MAT
# using GLMakie
using JLD2

# using CUDA
using JustPIC
using JustPIC._3D
# const backend = CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return range(xI, xF, length = n + 2)
end

A = matread("Taras/CornerFlow3D.mat")
const Vx_MAT = A["Vx"] |> TA(backend)
const Vy_MAT = A["Vy"] |> TA(backend)
const Vz_MAT = A["Vz"] |> TA(backend)

function main(np, integrator)

    Vx = Vx_MAT
    Vy = Vy_MAT
    Vz = Vz_MAT
    V = Vx, Vy, Vz

    lx = 100 # Horizontal model size, m
    ly = 100 # Vertical model size, m
    lz = 100 # Vertical model size, m
    nx = 40  # Horizontal grid resolution
    ny = 40  # Vertical grid resolution
    nz = 40  # Vertical grid resolution
    nx_v = nx + 1
    ny_v = ny + 1
    nz_v = nz + 1
    dx = lx / (nx) # Horizontal grid step, m
    dy = ly / (ny) # Vertical grid step, m
    dz = lz / (nz) # Vertical grid step, m
    x = range(0, lx, length = nx_v)  # Horizontal coordinates of basic grid points, m
    y = range(0, ly, length = ny_v)  # Vertical coordinates of basic grid points, m
    z = range(0, lz, length = nz_v)  # Vertical coordinates of basic grid points, m

    # nodal centers
    xc, yc, zc = range(0 + dx / 2, lx - dx / 2, length = nx), range(0 + dy / 2, ly - dy / 2, length = ny), range(0 + dz / 2, lz - dz / 2, length = nz)
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

    dt = min(dx / maximum(abs.(Array(Vx))), dy / maximum(abs.(Array(Vy))), dz / maximum(abs.(Array(Vz))))
    dt *= 0.75

    # ntime = 1000000
    ntime = 10000 #0
    ntime = 5000 #0
     np_Bi = zeros(Int64, ntime)
    np_MQS = zeros(Int64, ntime)
    np_LinP = zeros(Int64, ntime)
    # empty_Bi   = zeros(Int64, ntime)
    # empty_MQS  = zeros(Int64, ntime)
    # empty_LinP = zeros(Int64, ntime)
    # full_Bi    = zeros(Int64, ntime)
    # full_MQS   = zeros(Int64, ntime)
    # full_LinP  = zeros(Int64, ntime)

    injected_Bi = zeros(ntime)
    injected_MQS = zeros(ntime)
    injected_LinP = zeros(ntime)

    time_Bi = zeros(ntime)
    time_MQS = zeros(ntime)
    time_LinP = zeros(ntime)

    for it in 1:ntime
        time_Bi0 = @elapsed advection!(particles1, integrator, V, grid_vxi, dt)
        time_MQS0 = @elapsed advection_LinP!(particles2, integrator, V, grid_vxi, dt)
        time_LinP0 = @elapsed advection_MQS!(particles3, integrator, V, grid_vxi, dt)

        time_Bi0 += @elapsed move_particles!(particles1, xvi, ())
        time_MQS0 += @elapsed move_particles!(particles2, xvi, ())
        time_LinP0 += @elapsed move_particles!(particles3, xvi, ())


        np1 = sum(particles1.index.data)
        np2 = sum(particles2.index.data)
        np3 = sum(particles3.index.data)

        # for p in (particles1,particles2,particles3)
        #     inject_particles!(p, (), xvi)
        # end

        time_Bi0 += @elapsed inject_particles!(particles1, (), xvi)
        time_MQS0 += @elapsed inject_particles!(particles2, (), xvi)
        time_LinP0 += @elapsed inject_particles!(particles3, (), xvi)

        time_Bi[it] = time_Bi0
        time_MQS[it] = time_MQS0
        time_LinP[it] = time_LinP0

        np1 = sum(particles1.index.data) - np1
        np2 = sum(particles2.index.data) - np2
        np3 = sum(particles3.index.data) - np3

        # injected particles
        injected_Bi[it] = np1
        injected_LinP[it] = np2
        injected_MQS[it] = np3

        # total particles
        np_Bi[it] = sum(particles1.index.data)
        np_LinP[it] = sum(particles2.index.data)
        np_MQS[it] = sum(particles3.index.data)

        # # empty
        # push!(empty_Bi  , sum([all(iszero,p) for p in particles1.index]))
        # push!(empty_LinP, sum([all(iszero,p) for p in particles2.index]))
        # push!(empty_MQS , sum([all(iszero,p) for p in particles3.index]))

        # # full
        # push!(full_Bi  , sum([sum(isone, p) > np for p in particles1.index]))
        # push!(full_LinP, sum([sum(isone, p) > np for p in particles2.index]))
        # push!(full_MQS , sum([sum(isone, p) > np for p in particles3.index]))
    end
    stats_Bi = (; np = np_Bi, time = time_Bi, injected = injected_Bi)
    stats_LinP = (; np = np_LinP, time = time_MQS, injected = injected_LinP)
    stats_MQS = (; np = np_MQS, time = time_LinP, injected = injected_MQS)

    # stats_Bi   = (; np = np_Bi  , empty = empty_Bi  , time = time_Bi,   injected = injected_Bi, full = full_Bi)
    # stats_LinP = (; np = np_LinP, empty = empty_LinP, time = time_MQS,  injected = injected_LinP, full = full_LinP)
    # stats_MQS  = (; np = np_MQS , empty = empty_MQS , time = time_LinP, injected = injected_MQS, full = full_MQS)

    return particles1, particles2, particles3, stats_Bi, stats_LinP, stats_MQS
end

function runner()
    for integrator in (RungeKutta2(), RungeKutta4()), np in (4,)
        # for integrator in (RungeKutta2(), RungeKutta4()), np in (4,8,12,16,20,24)

        println("Sarting with np = $np...")
        particles1, particles2, particles3, stats_Lin, stats_LinP, stats_MQS = main(np, integrator)

        name = integrator == RungeKutta2() ? "RK2" : "RK4"

        jldsave(
            "Taras/data/CornerFlow3D_$(np)particles_injection_$(name).jld2",
            particles1 = particles1,
            particles2 = particles2,
            particles3 = particles3,
            stats_Lin = stats_Lin,
            stats_LinP = stats_LinP,
            stats_MQS = stats_MQS
        )
        println("...done with np = $np")

    end
    return
end

runner()

function main0()
    for np in (4, )
    # for np in (4, 8, 12, 16, 20, 24)

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

    return
end

main0()
