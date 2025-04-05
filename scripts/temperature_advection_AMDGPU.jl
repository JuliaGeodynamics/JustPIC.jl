using AMDGPU
using JustPIC
using JustPIC._2D

const backend = JustPIC.AMDGPUBackend
using CairoMakie

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
    nxcell, max_xcell, min_xcell = 24, 24, 24
    n = 256
    nx = ny = n - 1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = range(0, Lx, length = n), range(0, Ly, length = n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xc, yc = range(0 + dx / 2, Lx - dx / 2, length = n - 1), range(0 + dy / 2, Ly - dy / 2, length = n - 1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
    T = TA(backend)([y for x in xv, y in yv])
    V = Vx, Vy

    dt = min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)))

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1))
    grid2particle!(pT, xvi, T, particles)

    niter = 150
    for _ in 1:niter
        advection!(particles, RungeKutta2(2 / 3), V, (grid_vx, grid_vy), dt)
        move_particles!(particles, xvi, particle_args)
        particle2grid!(T, pT, xvi, particles)
    end

    f, ax, = heatmap(xvi..., Array(T), colormap = :batlow)
    streamplot!(ax, g, xvi...)
    return f
end

f = main()
