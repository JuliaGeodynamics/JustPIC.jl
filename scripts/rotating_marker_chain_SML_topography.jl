using JustPIC
using JustPIC._2D
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)
using GLMakie

const backend = JustPIC.CPUBackend

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return LinRange(xI, xF, n + 2)
end

# Analytical flow solution
vi_stream(x) = π * 1.0e-5 * (x - 0.5)

function main()

    # Initialize particles -------------------------------
    n = 51
    # nx = n - 1  # Removed unused variable
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dx, dy = xv[2] - xv[1], yv[2] - yv[1]  # Removed unused variable dxi
    # nodal centers
    xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv
    grid_vxi = grid_vx, grid_vy

    # Cell fields -------------------------------
    Vx = TA(backend)([-vi_stream(y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([ vi_stream(x) for x in grid_vy[1], y in grid_vy[2]])

    xc0 = yc0 = 0.25
    R = 24 * dx
    V = Vx, Vy

    w = π * 1.0e-5  # angular velocity
    period = 1  # revolution number
    tmax = period / (w / (2 * π))
    dt = 200.0

    nxcell, min_xcell, max_xcell = 12, 6, 24
    initial_elevation = fill(Ly / 2, length(xv))
    # Add Gaussian elevation to mimic a volcano
    volcano_center = 0.5
    volcano_height = 0.5
    volcano_width = 0.1
    gaussian_elevation = volcano_height .* exp.(-((xv .- volcano_center) .^ 2) ./ (2 * volcano_width^2))
    initial_elevation = Ly / 2
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, initial_elevation)
    steep_topography = Ly / 2 .+ gaussian_elevation
    fill_chain_from_vertices!(chain, steep_topography)
    method = RungeKutta4()

    for _ in 1:125
        semilagrangian_advection_markerchain!(chain, method, V, grid_vxi, xvi, dt; max_slope_angle = 45.0)
    end

    f = Figure(size = (1200, 1200))
    ax = Axis(f[1, 1])
    # vector of shapes
    poly!(
        ax,
        Rect(0, 0, 1, 1),
        color = :lightgray,
    )
    lines!(ax, xvi[1], chain.h_vertices, color = :blue, linewidth = 4)
    display(f)
    return nothing
end

main()
