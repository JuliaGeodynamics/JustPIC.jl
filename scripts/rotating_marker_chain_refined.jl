using JustPIC
using GLMakie

const backend = JustPIC.CPU

include(joinpath(@__DIR__, "refined_grid_utils.jl"))

# Analytical flow solution
vi_stream(x) = π * 1.0e-5 * (x - 0.5)

function main()

    # Initialize particles -------------------------------
    n = 51
    nx = ny = n - 1
    Lx = Ly = 1.0
    # graded nodal vertices, five times finer at the center than a uniform grid would be
    xv, xc, dx = makeExpoGrid(Lx, nx, Lx / nx / 5, 0.0)
    yv, yc, dy = makeExpoGrid(Ly, ny, Ly / ny / 5, 0.0)
    # staggered grid velocity nodal locations
    yv_vx = expand_range(yc)
    xv_vy = expand_range(xc)

    # a refined grid is indexed directly inside the kernels, so it lives on the device
    xvi = TA(backend)(xv), TA(backend)(yv)
    grid_vx = TA(backend)(xv), TA(backend)(yv_vx)
    grid_vy = TA(backend)(xv_vy), TA(backend)(yv)
    grid_vxi = grid_vx, grid_vy

    # Cell fields -------------------------------
    Vx = TA(backend)([-vi_stream(y) for x in xv, y in yv_vx])
    Vy = TA(backend)([ vi_stream(x) for x in xv_vy, y in yv])
    V = Vx, Vy

    dt = 200.0

    nxcell, min_xcell, max_xcell = 12, 6, 24
    initial_elevation = Ly / 2
    chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xvi[1], initial_elevation)
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
    vlines!(ax, xv, color = (:gray, 0.5))
    px = chain.coords[1].data[:]
    py = chain.coords[2].data[:]
    scatter!(px, py, color = :black)
    return display(f)

end

main()
