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
    # nxcell, max_xcell, min_xcell = 9, 9, 1
    n = 51
    nx = n - 1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
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
    T = TA(backend)([ ((x - xc0)^2 + (y - yc0)^2 ≤ R^2) * 1.0 for x in xv, y in yv])
    V = Vx, Vy

    w = π * 1.0e-5  # angular velocity
    period = 1  # revolution number
    tmax = period / (w / (2 * π))
    dt = 200.0

    nxcell, min_xcell, max_xcell = 12, 6, 24
    initial_elevation = Ly / 2
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, initial_elevation)
    method = RungeKutta2()

    for _ in 1:25
        advect_markerchain!(chain, method, V, grid_vxi, dt)
    end

    f = Figure()
    ax = Axis(f[1, 1])
    # vector of shapes
    poly!(
        ax,
        Rect(0, 0, 1, 1),
        color = :lightgray,
    )
    px = chain.coords[1].data[:]
    py = chain.coords[2].data[:]
    scatter!(px, py, color = :black)
    return display(f)

end

n = 51
nx = n - 1
Lx = Ly = 1.0
# nodal vertices
xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
# nodal centers
xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1)
# staggered grid velocity nodal locations
grid_vx = xv, expand_range(yc)
grid_vy = expand_range(xc), yv
grid_vxi = grid_vx, grid_vy

# Cell fields -------------------------------
Vx = TA(backend)([-vi_stream(y) for x in grid_vx[1], y in grid_vx[2]]);
Vy = TA(backend)([ vi_stream(x) for x in grid_vy[1], y in grid_vy[2]]);

nxcell, min_xcell, max_xcell = 12, 6, 24
initial_elevation = Ly / 2
chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, initial_elevation);

chain_V = similar(chain.coords[1]), similar(chain.coords[1]);
chain_V[1].data .*= 0
chain_V[2].data .*= 0

interpolate_velocity_to_markerchain!(chain, chain_V, V, grid_vxi)

a, r, t1, t2 = rand(4)

f1(a, r, t1, t2) = exp(a / (r * t1) - a / (r * t2))
f2(a, r, t1, t2) = exp((a * (t2 - t1)) / (r * t1 * t2))

@be f1($(a, r, t1, t2)...)
@be f2($(a, r, t1, t2)...)

@code_native f1(a, r, t1, t2)
@code_native f2(a, r, t1, t2)
@code_llvm f3(a, r, t1, t2)
