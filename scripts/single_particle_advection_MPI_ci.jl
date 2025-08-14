# using CUDA
using JustPIC, JustPIC._2D

using ImplicitGlobalGrid
import MPI

# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"),
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
# const backend = CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
const backend = JustPIC.CPUBackend

using ParallelStencil
# @init_parallel_stencil(CUDA, Float64, 2)
@init_parallel_stencil(Threads, Float64, 2)

# using GLMakie

@inline init_particle_fields_cellarrays(particles, ::Val{N}) where {N} = ntuple(_ -> @fill(0.0, size(particles.coords[1])..., celldims = (cellsize(particles.index))), Val(N))

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return LinRange(xI, xF, n + 2)
end

function expand_range(x::AbstractArray, dx)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return x = TA(backend)(vcat(xI, x, xF))
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
    nxcell, max_xcell, min_xcell = 24, 24, 18
    n = 32
    nx = ny = n - 1
    me, dims, = init_global_grid(
        n - 1, n - 1, 1;
        init_MPI = JustPIC.MPI.Initialized() ? false : true
    )
    Lx = Ly = 1.0
    dxi = dx, dy = Lx / (nx_g() - 1), Ly / (ny_g() - 1)
    # nodal vertices
    xvi = xv, yv = let
        dummy = zeros(n, n)
        xv = TA(backend)([x_g(i, dx, dummy) for i in axes(dummy, 1)])
        yv = TA(backend)([y_g(i, dx, dummy) for i in axes(dummy, 2)])
        xv, yv
    end
    # nodal centers
    xci = xc, yc = let
        dummy = zeros(nx, ny)
        xc = TA(backend)([x_g(i, dx, dummy) for i in axes(dummy, 1)])
        yc = TA(backend)([y_g(i, dx, dummy) for i in axes(dummy, 2)])
        xc, yc
    end

    # staggered grid for the velocity components
    grid_vx = xv, expand_range(yc, dy)
    grid_vy = expand_range(xc, dx), yv

    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xv, yv, dxi..., nx, ny, me
    )

    # allocate particle field
    particle_args = ()

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
    V = Vx, Vy

    # time step
    dt = min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)))

    nx_v = (size(particles.coords[1].data, 2)) * dims[1]
    ny_v = (size(particles.coords[1].data, 3) - 2) * dims[2]
    px_v = fill(NaN, nx_v, ny_v)
    py_v = fill(NaN, nx_v, ny_v)
    index_v = fill(false, nx_v, ny_v)
    px_nohalo = fill(NaN, size(particles.coords[1].data, 2), size(particles.coords[1].data, 3) - 2)
    py_nohalo = fill(NaN, size(particles.coords[1].data, 2), size(particles.coords[1].data, 3) - 2)
    index_nohalo = fill(false, size(particles.coords[1].data, 2), size(particles.coords[1].data, 3) - 2)

    p = [(NaN, NaN)]

    # Advection test
    niter = 150
    for iter in 1:niter

        # advect particles
        advection!(particles, RungeKutta2(2 / 3), V, (grid_vx, grid_vy), dt)
        # update halos
        update_cell_halo!(particles.coords..., particle_args...)
        update_cell_halo!(particles.index)

        # shuffle particles
        move_particles!(particles, xvi, particle_args)

        # gather particle data - for plotting only
        @views px_nohalo .= particles.coords[1].data[1, :, 2:(end - 1)]
        @views py_nohalo .= particles.coords[2].data[1, :, 2:(end - 1)]
        @views index_nohalo .= particles.index.data[1, :, 2:(end - 1)]
        gather!(px_nohalo, px_v)
        gather!(py_nohalo, py_v)
        gather!(index_nohalo, index_v)

        if me == 0
            p_i = (px_v[index_v][1], py_v[index_v][1])
            push!(p, p_i)
        end

        # if me == 0 && iter % 10 == 0
        #     w = 0.504
        #     offset = 0.5 - (w - 0.5)
        #     f, ax, = lines(
        #         [0, w, w, 0, 0],
        #         [0, 0, w, w, 0],
        #         linewidth = 3
        #     )
        #     lines!(
        #         ax,
        #         [0, w, w, 0, 0] .+ offset,
        #         [0, 0, w, w, 0],
        #         linewidth = 3
        #     )
        #     lines!(
        #         ax,
        #         [0, w, w, 0, 0] .+ offset,
        #         [0, 0, w, w, 0] .+ offset,
        #         linewidth = 3
        #     )
        #     lines!(
        #         ax,
        #         [0, w, w, 0, 0],
        #         [0, 0, w, w, 0] .+ offset,
        #         linewidth = 3
        #     )
        #     streamplot!(ax, g, LinRange(0, 1, 100), LinRange(0, 1, 100))
        #     lines!(ax, p, color = :red)
        #     scatter!(ax, p[end], color = :black)
        #     save("figs/trajectory_MPI_$iter.png", f)
        # end
    end

    return finalize_global_grid()
end

main()
