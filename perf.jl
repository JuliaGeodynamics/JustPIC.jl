const isGPU = true

@static if isGPU
    using CUDA
end

using JustPIC, JustPIC._2D 
const backend = @static if isGPU
    CUDABackend
else
    JustPIC.CPUBackend
end

using ParallelStencil, StaticArrays, LinearAlgebra, ParallelStencil, TimerOutputs
using CSV, DataFrames
# using GLMakie

@static if isGPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end


function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    range(xI, xF, length=n+2)
end

# Analytical flow solution
vx_stream(x, y) =  250 * sin(π*x) * cos(π*y)
vy_stream(x, y) = -250 * cos(π*x) * sin(π*y)

function advection_test_2D(n, np)
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = np, np, np
    # n = 64
    n += 1
    nx = ny = n-1
    ni = nx , ny
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = range(0, Lx, length=n), range(0, Ly, length=n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xci = xc, yc = range(0+dx/2, Lx-dx/2, length=n-1), range(0+dy/2, Ly-dy/2, length=n-1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi, dxi, (nx, ny)
    )

    # passive_coords = ntuple(Val(2)) do i
    #     rand(np*nx*ny) .* 0.90 .+ 0.05
    # end

    xp = filter(!isnan, particles.coords[1].data)
    yp = filter(!isnan, particles.coords[2].data)
    # passive_coords = [@SVector [xp[i],yp[i]] for i in 1:np*nx*ny]
    passive_coords = xp, yp

    # passive_coords = [(@SVector rand(2)).* 0.90 .+ 0.05 for i in 1:np*nx*ny]
    passive_markers = PassiveMarkers(backend, passive_coords);
    # passive_markers = init_passive_markers(backend, passive_coords);
    T_marker = @zeros(np*nx*ny)

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]]);
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]]);
    T  = TA(backend)([y for x in xv, y in yv]);
    T0 = deepcopy(T)
    buffer = similar(T)
    V  = Vx, Vy;

    dt = min(dx / maximum(abs.(Array(Vx))),  dy / maximum(abs.(Array(Vy)))) / 2;
    particle_args = pT0, = init_cell_arrays(particles, Val(1));
    # pT = @zeros(length(passive_markers.coords))
    # x_copy = copy(particles.coords)

    niter = 50
    t = 0.0

    time_opt         = Float64[]
    time_classic     = Float64[]
    time_adv_opt     = Float64[]
    time_injection   = Float64[]
    time_shuffle     = Float64[]
    time_adv_classic = Float64[]
    time_g2p_opt     = Float64[]
    time_p2g_opt     = Float64[]
    time_g2p_classic = Float64[]
    time_p2g_classic = Float64[]

    it = 0 
    grid_vxi = grid_vx, grid_vy
    # while it < 100
    while t < 0.04 / 2
        it += 1
        push!(time_opt, @elapsed begin        
            push!(time_adv_opt, @elapsed advection!(particles, RungeKutta2(2/3), V, grid_vxi, dt))
            # push!(time_adv_opt, @elapsed advection_RK!(particles, V, grid_vx, grid_vy, dt, 2 / 3))
            push!(time_shuffle, @elapsed move_particles!(particles, xvi, particle_args))
            push!(time_injection, @elapsed inject_particles!(particles, (pT0, ), xvi))
            push!(time_g2p_opt, @elapsed grid2particle!(pT0, xvi, T, particles))
            push!(time_p2g_opt, @elapsed particle2grid!(T, pT0, xvi, particles))

        end)
        push!(time_classic, @elapsed begin
            push!(time_adv_classic, @elapsed advection!(passive_markers, RungeKutta2(), V, (grid_vx, grid_vy), dt))
            push!(time_g2p_classic, @elapsed grid2particle!(T_marker, xvi, T, passive_markers) )
            push!(time_p2g_classic, @elapsed particle2grid!(T0, T_marker, buffer, xvi, passive_markers))
        end)
        t += dt
    end

    time_opt .-= time_injection

    timings = (
        opt = time_opt,
        classic = time_classic,
        adv_opt = time_adv_opt,
        shuffle = time_shuffle,
        adv_classic = time_adv_classic,
        g2p_opt = time_g2p_opt,
        p2g_opt = time_p2g_opt,
        g2p_classic = time_g2p_classic,
        p2g_classic = time_p2g_classic,
    )
    return timings
end

# function run_cuda()
#     n = (64,) #128, 256#, 512# 1024
#     n = 32, 64, #128, 256#, 512# 1024
#     # n = 512, 1024
#     np = 6, 12, 18, 24
#     fldr = "timings_CUDA"
#     !isdir(fldr) && mkdir(fldr)
#     for ni in n, npi in np
#         println("Running n = $ni, np = $npi")
#         timings = advection_test_2D(ni, npi)
#         dt = DataFrame(timings)
#         CSV.write("timings_CUDA/timings_n$(ni)_np$(npi).csv", dt)
#     end
# end

function run_cpu()
    n = 32, 64, 128, 256
    np = 6, 12, 18, 24
    nt = Threads.nthreads()
    fldr = "timings_cpu_new/nt$(nt)"
    !isdir(fldr) && mkpath(fldr)
    !isdir(fldr) && mkdir(fldr)
    for ni in n, npi in np
        println("Running n = $ni, np = $npi")
        timings = advection_test_2D(ni, npi)
        dt = DataFrame(timings)
        CSV.write(joinpath(fldr, "CUDA_timings_n$(ni)_np$(npi).csv"), dt)
    end
end


function run_gpu()
    n  = 128, 256, 512, 1024#, 2048
    np = 6, 12#, 18, 24

    fldr = "timings_gpu/"
    !isdir(fldr) && mkpath(fldr)
    !isdir(fldr) && mkdir(fldr)

    for ni in n, npi in np
        println("Running n = $ni, np = $npi")
        timings = advection_test_2D(ni, npi)
        dt = DataFrame(timings)
        CSV.write(joinpath(fldr, "CUDA_timings_n$(ni)_np$(npi).csv"), dt)
    end
end

@static if isGPU
    run_gpu()
else
    run_cpu()
end


