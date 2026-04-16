# using CUDA
using JustPIC
using JustPIC._2D

# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"),
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
# const backend = CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

using GLMakie

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return LinRange(xI, xF, n + 2)
end

# Analytical flow solution
vx_stream(x, y) = 1.0
vy_stream(x, y) = 0.0
g(x) = Point2f(
    vx_stream(x[1], x[2]),
    vy_stream(x[1], x[2])
)

@inline incircles(x, y, xc, yc, r1, r2) = r1^2 ≤ (x - xc)^2 + (y - yc)^2 ≤ r2^2

function main()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 30, 12
    n = 256
    nx = ny = n - 1
    Lx = 5.0e0
    Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xci = xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv
    grid_vel = grid_vx, grid_vy
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, grid_vel...,
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
    xc, yc, r1, r2 = 1.0, 0.5, 0.1, 0.3
    T = TA(backend)([incircles(x, y, xc, yc, r1, r2) * 1.0e0 for x in particles.xvi[1], y in particles.xvi[2]])
    # T = TA(backend)([incircles(x, y, xc, yc, r1, r2) * 1.0e0 for x in xv, y in yv])
    V = Vx, Vy

    dt = 0.018

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1))
    grid2particle!(pT, T, particles)
    # grid2particle!(pT, T, particles; ghost_1 = false, ghost_2 = false)

    frame_stride = 10
    t_end = 5.0
    framerate = 30
    animation_file = "donut_advection_periodic_dt_$(dt).gif"

    t = 0.0
    it = 0

    title = Observable("Donut Advection - Δt = $dt, t = $(round(t; digits = 3))")
    T_frame = Observable(Array(T))

    f = Figure(size = (800, 400) .* 2)
    ax = Axis(f[1, 1], aspect = 5, title = title, xlabel = "x", ylabel = "y")
    heatmap!(ax, xvi..., T_frame, colormap = :batlow)
    # streamplot!(ax, g, xvi...)

    nsteps = ceil(Int, t_end / dt)
    nframes = cld(nsteps, frame_stride)
    record(f, animation_file, 0:nframes; framerate = framerate) do frame
        if frame > 0
            for _ in 1:frame_stride
                t ≥ t_end && break
                @show it += 1
                advection!(particles, RungeKutta2(), V, dt)
                move_particles!(particles, particle_args; periodic_1 = true, periodic_2 = false)
                inject_particles!(particles, (pT,))
                particle2grid!(T, pT, particles)
                # particle2grid!(T, pT, particles; ghost_1 = false, ghost_2 = false)
                any(isnan, T)
                t += dt
            end
        end
        T_frame[] = Array(T)
        title[] = "Donut Advection - Δt = $dt, t = $(round(t; digits = 3))"
    end

    return println("Finished writing $(animation_file)")
end

main()
