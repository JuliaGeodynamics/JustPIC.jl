using JustPIC

# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"),
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = JustPIC.CPU # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend

# using GLMakie

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return LinRange(xI, xF, n + 2)
end

# Analytical flow solution
vx_stream(x, y) = 250 * sin(pi * x) * cos(pi * y)
vy_stream(x, y) = -250 * cos(pi * x) * sin(pi * y)

function timed!(label, f)
    t = @elapsed f()
    println(label, ": ", round(t; digits = 6), " s")
    return t
end

function main()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 48, 28
    n = parse(Int, get(ENV, "JUSTPIC_TIMER_N", "256"))
    nx = ny = n - 1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
    T = TA(backend)([y for x in xv, y in yv])
    V = Vx, Vy

    dt = min(dx / maximum(abs.(Array(Vx))), dy / maximum(abs.(Array(Vy))))
    dt *= 0.25

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1))
    grid2particle!(pT, T, particles)

    niter = parse(Int, get(ENV, "JUSTPIC_TIMER_NITER", "5"))
    for it in 1:niter
        println("iteration ", it)
        timed!("advect", () -> advection!(particles, RungeKutta2(2 / 3), V, dt))
        timed!("move", () -> move_particles!(particles, particle_args))
        timed!("injection", () -> inject_particles!(particles, (pT,)))
        timed!("p2g", () -> particle2grid!(T, pT, particles))
        timed!("g2p", () -> grid2particle!(pT, T, particles))
    end

    return println("Finished")
end

main()
