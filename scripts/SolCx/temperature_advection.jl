using JustPIC
using JustPIC._2D

# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"),
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

using GLMakie

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1 - dx; sigdigits = 5)
    xF = round(x2 + dx; sigdigits = 5)
    return range(xI, xF, length = n + 2)
end

# Analytical flow solution
include("SolCx_solution.jl")

function main()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 25, 60, 2
    n = 32
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
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )

    Vx = [_solCx_solution(x, y, 1, 1.0e4)[1] for x in grid_vx[1], y in grid_vx[2]]
    Vy = [_solCx_solution(x, y, 1, 1.0e4)[2] for x in grid_vy[1], y in grid_vy[2]]

    # Cell fields -------------------------------
    T = TA(backend)([y for x in xv, y in yv])
    V = Vx, Vy

    dt = min(dx / maximum(abs.(Array(Vx))), dy / maximum(abs.(Array(Vy))))
    dt *= 0.1
    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1))
    grid2particle!(pT, xvi, T, particles)

    !isdir("figs") && mkdir("figs")

    niter = 1000
    for it in 1:niter
        advection!(particles, RungeKutta2(), V, (grid_vx, grid_vy), dt)
        move_particles!(particles, xvi, particle_args)
        inject_particles!(particles, (pT,), xvi)
        particle2grid!(T, pT, xvi, particles)

        if rem(it, 10) == 0
            f, ax, = heatmap(xvi..., Array(T), colormap = :batlow)
            streamplot!(ax, g, xvi...)
            save("figs/test_$(it).png", f)
            f
        end
    end
    return println("Finished")
end

main()
