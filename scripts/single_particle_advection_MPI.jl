using JustPIC
using JustPIC._2D

# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"),
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

using GLMakie
using ImplicitGlobalGrid
using MPI: MPI

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return LinRange(xI, xF, n + 2)
end

# Analytical flow solution
vx_stream(x, y) = 250 * sin(π * x) * cos(π * y)
vy_stream(x, y) = -250 * cos(π * x) * sin(π * y)
g(x) = Point2f(
    vx_stream(x[1], x[2]),
    vy_stream(x[1], x[2])
)

function main()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 40, 1
    n = 64
    nx = ny = n - 1
    me, dims, = init_global_grid((nx, ny) .+ 1..., 1; init_MPI = MPI.Initialized() ? false : true)
    Lx = Ly = 1.0
    dxi = dx, dy = Lx / (nx_g() - 1), Ly / (ny_g() - 1)
    # nodal vertices
    xvi = xv, yv = let
        dummy = zeros(n, n)
        xv = [x_g(i, dx, dummy) for i in axes(dummy, 1)]
        yv = [y_g(i, dy, dummy) for i in axes(dummy, 2)]
        LinRange(first(xv), last(xv), n), LinRange(first(yv), last(yv), n)
    end
    # nodal centers
    xci = xc, yc = let
        dummy = zeros(nx, ny)
        xc = [x_g(i, dx, dummy) for i in axes(dummy, 1)]
        yc = [y_g(i, dy, dummy) for i in axes(dummy, 2)]
        LinRange(first(xc), last(xc), nx), LinRange(first(yc), last(yc), ny)
    end

    # staggered grid for the velocity components
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy
    )

    particle_args = ()

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
    V = Vx, Vy

    dt = mapreduce(x -> x[1] / MPI.Allreduce(maximum(abs.(x[2])), MPI.MAX, MPI.COMM_WORLD), min, zip(dxi, V)) / 2

    nx_v = (size(particles.coords[1].data, 2) - 2) * dims[1]
    ny_v = (size(particles.coords[1].data, 3) - 2) * dims[2]
    px_v = fill(NaN, nx_v, ny_v)
    py_v = fill(NaN, nx_v, ny_v)
    index_v = fill(false, nx_v, ny_v)
    px_nohalo = fill(NaN, size(particles.coords[1].data, 2) - 2, size(particles.coords[1].data, 3) - 2)
    py_nohalo = fill(NaN, size(particles.coords[1].data, 2) - 2, size(particles.coords[1].data, 3) - 2)
    index_nohalo = fill(false, size(particles.coords[1].data, 2) - 2, size(particles.coords[1].data, 3) - 2)

    p = [(NaN, NaN)]
    !isdir("figs") && mkdir("figs")

    # Advection test
    niter = 150
    for iter in 1:niter

        # advect particles
        advection!(particles, RungeKutta2(), V, dt)
        # update halos
        update_cell_halo!(particles.coords..., particles.index)

        # shuffle particles
        move_particles!(particles, particle_args)

        # gather particle data - for plotting only
        @views px_nohalo .= particles.coords[1].data[1, 2:(end - 1), 2:(end - 1)]
        @views py_nohalo .= particles.coords[2].data[1, 2:(end - 1), 2:(end - 1)]
        @views index_nohalo .= particles.index.data[1, 2:(end - 1), 2:(end - 1)]
        gather!(px_nohalo, px_v)
        gather!(py_nohalo, py_v)
        gather!(index_nohalo, index_v)

        if me == 0 && any(index_v)
            p_i = (px_v[index_v][1], py_v[index_v][1])
            push!(p, p_i)
        end

        if me == 0 && iter % 10 == 0
            w = 0.504
            offset = 0.5 - (w - 0.5)
            f, ax, = lines(
                [0, w, w, 0, 0],
                [0, 0, w, w, 0],
                linewidth = 3
            )
            lines!(
                ax,
                [0, w, w, 0, 0] .+ offset,
                [0, 0, w, w, 0],
                linewidth = 3
            )
            lines!(
                ax,
                [0, w, w, 0, 0] .+ offset,
                [0, 0, w, w, 0] .+ offset,
                linewidth = 3
            )
            lines!(
                ax,
                [0, w, w, 0, 0],
                [0, 0, w, w, 0] .+ offset,
                linewidth = 3
            )
            streamplot!(ax, g, LinRange(0, 1, 100), LinRange(0, 1, 100))
            lines!(ax, p, color = :red)
            scatter!(ax, p[end], color = :black)
            save("figs/trajectory_MPI_$iter.png", f)
        end
    end

    return finalize_global_grid()
end

main()
