using JustPIC, JustPIC._2D 
const backend = CPUBackend

using ParallelStencil, StaticArrays, LinearAlgebra, ParallelStencil, TimerOutputs
using CSV, DataFrames
using GLMakie

@init_parallel_stencil(Threads, Float64, 2)

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
    nxcell, max_xcell, min_xcell = np, np*2, 1
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
        rand(np*nx*ny) .* 0.9 .+ 0.05
    end
    passive_markers = init_passive_markers(backend, passive_coords);
    T_marker = zeros(np*nx*ny)

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
    time_shuffle     = Float64[]
    time_adv_classic = Float64[]
    time_g2p_opt     = Float64[]
    time_p2g_opt     = Float64[]
    time_g2p_classic = Float64[]
    time_p2g_classic = Float64[]

    it = 0 
    while it < 10
    # while t < 0.04
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

    # np_tot = sum(sum(p) for p in particles0.index)
    # println("Number of particles: ", np_tot)
    # niter = 2000
end

function run_cuda()
    n = 64, 128, 256#, 512# 1024
    # n = 512, 1024
    np = 6, 12, 18, 24
    fldr = "timings_CUDA"
    !isdir(fldr) && mkdir(fldr)
    for ni in n, npi in np
        println("Running n = $ni, np = $npi")
        timings = advection_test_2D(ni, npi)
        dt = DataFrame(timings)
        CSV.write("timings_CUDA/timings_n$(ni)_np$(npi).csv", dt)
    end
end

function run_cpu()
    n = 32, 64 , 128
    n = (256,)
    np = 6, 12, 18, 24
    # np = (24,)
    nt = Threads.nthreads()
    fldr = "timings_cpu/nt$(nt)"
    !isdir(fldr) && mkpath(fldr)
    !isdir(fldr) && mkdir(fldr)
    for ni in n, npi in np
        println("Running n = $ni, np = $npi")
        timings = advection_test_2D(ni, npi)
        dt = DataFrame(timings)
        CSV.write(fldr*"/timings_n$(ni)_np$(npi).csv", dt)
    end
end

# run_cuda()
run_cpu()
1
# popfirst!(timings.opt)
# popfirst!(timings.classic)
# f, ax, = scatter(timings.opt, color=:black)
# scatter!(ax, timings.classic, color=:red)
# f

# x     = 1:length(timings.classic)
# lr    = linregress(x, timings.classic./timings.opt)
# f,ax, = scatter(x, timings.classic./timings.opt, color=:black)
# lines!(x, lr.coeffs[1] .* x .+lr.coeffs[2], color=:red)
# f

# p = particles.coords
# parent = [LinearIndices((1:nx, 1:ny))[get_cell(p, dxi)...] for p in p]
# max_dist = [max(parent[i]-parent[i-1], parent[i]-parent[i-1]) for i in 2:length(p)-1]
# max_dist = vcat([parent[1]-parent[2]], max_dist, parent[end]-parent[end-1])
# max_dist2 = max_dist .< nx
# scatter(p, color=max_dist, markersize=10)

# f,ax,s=scatter(
#     p, 
#     color=log10.(abs.(max_dist)), 
#     markersize=2,
#     colormap=:managua
# )
# xlims!(ax, 0, 1); ylims!(ax, 0, 1);
# Colorbar(f[1,2], s); f

# f,ax,s=scatter(
#     p, 
#     color=max_dist2, 
#     markersize=2,
#     colormap=:grayC
# )
# xlims!(ax, 0, 1); ylims!(ax, 0, 1);
# Colorbar(f[1,2], s); f

# ProfileCanvas.@profview for i in 1:100
#     grid2particle!(pT0, xvi, T, particles0.coords, particles0.index)
# end
# ProfileCanvas.@profview for i in 1:100
#     grid2particle_naive!(pT, xvi, T, particles)
# end


# # normalize coordinates
# @inline function f1(
#     p::Union{SVector{N,A}, NTuple{N,A}}, xi::NTuple{N,B}, di::NTuple{N,C}, idx::NTuple{N,D}
# ) where {N,A,B,C,D}
#     return ntuple(Val(N)) do i
#         Base.@_inline_meta
#         @inbounds (p[i] - xi[i][idx[i]]) 
#         # @inbounds (p[i] - xi[i][idx[i]]) * inv(di[i])
#     end
# end

# @generated function f2(
#     p::NTuple{N,A}, xi::NTuple{N,B}, di::NTuple{N,C}, idx::NTuple{N,D}
# ) where {N,A,B,C,D}
#     quote 
#         Base.@_inline_meta
#         Base.@nexprs $N i -> x_i = @inbounds (p[i] - xi[i][idx[i]]) * inv(di[i])
#         Base.@ncall $N tuple x
#     end
# end

# @btime f2($(p, xi, di, idx)...)
# @code_warntype f2(p, xi, di, idx)

# ProfileCanvas.@profview for i in 1:1000000
#     f2(p, xi, di, idx)
#     # f1(p, xi, di, idx)
# end
