using CUDA
using JustPIC, JustPIC._2D 
const backend = CUDABackend

using ParallelStencil
using StaticArrays, LinearAlgebra, ParallelStencil, TimerOutputs, CSV, DataFrames, GLMakie

@init_parallel_stencil(CUDA, Float64, 2)

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

n, np= 128, 12

function advection_test_2D(n, np)
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = np, np, 1
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
        backend, nxcell, max_xcell, min_xcell, xvi..., dxi..., nx, ny
    )

    passive_coords = ntuple(Val(2)) do i
        @rand(np*nx*ny) .* 0.9 .+ 0.05
    end
    passive_markers = init_passive_markers(backend, passive_coords);
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
    pT = @zeros(length(passive_markers.coords))
    # x_copy = copy(particles.coords)

    niter = 50
    t = 0.0

    time_opt         = Float64[]
    time_classic     = Float64[]
    time_adv_opt     = Float64[]
    time_shuffle     = Float64[]
    time_adv_classic = Float64[]
    time_g2p_opt     = Float64[]
    time_p2g_opt     = Float64[]
    time_g2p_classic = Float64[]
    time_p2g_classic = Float64[]

    it = 0 
    # while it < 10
    while t < 0.04
        it += 1
        push!(time_opt, @elapsed begin
            push!(time_adv_opt, @elapsed advection_RK!(particles, V, grid_vx, grid_vy, dt, 2 / 3))
            # push!(time_shuffle, @elapsed shuffle_particles!(particles0, xvi, particle_args))
            push!(time_shuffle, @elapsed move_particles!(particles, xvi, particle_args))
            push!(time_g2p_opt, @elapsed grid2particle!(pT0, xvi, T, particles))
            push!(time_p2g_opt, @elapsed particle2grid!(T, pT0, xvi, particles))

        end)
        push!(time_classic, @elapsed begin
            push!(time_adv_classic, @elapsed advect_passive_markers!(passive_markers, V, grid_vx, grid_vy, dt))
            push!(time_g2p_classic, @elapsed grid2particle!(T_marker, xvi, T, passive_markers) )
            push!(time_p2g_classic, @elapsed particle2grid!(T0, T_marker, buffer, xvi, passive_markers))
        end)
        t += dt
    end

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

function run_cuda()
    n = 64, 128, 256#, 512# 1024
    # n = (64,) 
    # n = 512, 1024
    np = 6, 12, 18, 24
    # np = (6,)
    fldr = "timings_CUDA"
    !isdir(fldr) && mkdir(fldr)
    for ni in n, npi in np
        println("Running n = $ni, np = $npi")
        timings = advection_test_2D(ni, npi)
        dt = DataFrame(timings)
        CSV.write("timings_CUDA/timings_n$(ni)_np$(npi).csv", dt)
    end
end

# run_cuda()


advection_RK!(particles, V, grid_vx, grid_vy, dt, 2 / 3)
move_particles!(particles, xvi, particle_args)
grid2particle!(pT0, xvi, T, particles)
particle2grid!(T, pT0, xvi, particles)

advect_passive_markers!(passive_markers, V, grid_vx, grid_vy, dt)
grid2particle!(T_marker, xvi, T, passive_markers)
particle2grid!(T0, T_marker, buffer, xvi, passive_markers)


xpm = Array(passive_markers.coords[1].data[:])
ypm = Array(passive_markers.coords[2].data[:])

scatter(xpm, ypm, color=Array(T_marker))


@btime particle2grid!($(T, pT0, xvi, particles)...)
@btime particle2grid!($(T0, T_marker, buffer, xvi, passive_markers)...)
