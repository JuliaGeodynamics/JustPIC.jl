using JustPIC, JustPIC._2D

# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"),
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

using GLMakie
using ImplicitGlobalGrid
using MPI: MPI

# Analytical flow solution
vx_stream(x, y) = 250 * sin(π * x) * cos(π * y)
vy_stream(x, y) = -250 * cos(π * x) * sin(π * y)
g(x) = Point2f(
    vx_stream(x[1], x[2]),
    vy_stream(x[1], x[2])
)

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return LinRange(xI, xF, n + 2)
end

function main()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 40, 1
    n = 64 # number of vertices
    nx = ny = n - 1
    me, dims, = init_global_grid((nx, ny).+1..., 1; init_MPI = MPI.Initialized() ? false : true)
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

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
    xvi_p = Array.(particles.xvi)
    T = TA(backend)([y for x in xvi_p[1], y in xvi_p[2]])
    T0 = deepcopy(T)
    V = Vx, Vy

    nx_v = (size(T, 1) - 2) * dims[1]
    ny_v = (size(T, 2) - 2) * dims[2]
    T_v = zeros(nx_v, ny_v)
    T_nohalo = TA(backend)(zeros(size(T) .- 2))

    dt = mapreduce(x -> x[1] / MPI.Allreduce(maximum(abs.(x[2])), MPI.MAX, MPI.COMM_WORLD), min, zip(dxi, V)) / 2

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1))
    grid2particle!(pT, T, particles)

    !isdir("figs") && mkdir("figs")

    niter = 250
    for iter in 1:niter
        me == 0 && @show iter

        # advect particles
        advection!(particles, RungeKutta2(), V, dt)
        
        # update halos
        timer += @elapsed begin
            update_cell_halo!(particles.coords..., particle_args..., particles.index)
        end
        # shuffle particles
        move_particles!(particles, particle_args)
        # refill under-populated cells before reconstructing the grid field
        # inject_particles!(particles, particle_args)
        # interpolate T from particle to grid
        particle2grid!(T, pT, particles)

        @views T_nohalo .= T[2:(end - 1), 2:(end - 1)]
        gather!(Array(T_nohalo), T_v)

        if me == 0 && iter % 1 == 0
            x_global = LinRange(0, Lx, size(T_v, 1))
            y_global = LinRange(0, Ly, size(T_v, 2))
            f, ax, = heatmap(x_global, y_global, T_v)
            w = 0.504
            offset = 0.5 - (w - 0.5)
            lines!(
                ax,
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

            save("figs/T_MPI_$iter.png", f)
        end

        # px = particles.coords[1].data[:]
        # py = particles.coords[2].data[:]
        # idx = particles.index.data[:]
        # f = scatter(px[idx], py[idx], color=:black)
        # save("figs/particles_$(iter)_$(me).png", f)
    end

    iszero(me) && @show timer

    # f, ax, = heatmap(xvi..., T, colormap=:batlow)
    # streamplot!(ax, g, xvi...)
    # f
    return finalize_global_grid()

end

main()