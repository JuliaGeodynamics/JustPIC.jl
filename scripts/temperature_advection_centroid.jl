using CUDA
using JustPIC, JustPIC._2D

# Threads is the default backend, 
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"), 
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
# const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
const backend = CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

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
vx_stream(x, y) =  250 * sin(π*x) * cos(π*y)
vy_stream(x, y) = -250 * cos(π*x) * sin(π*y)
g(x) = Point2f(
    vx_stream(x[1], x[2]),
    vy_stream(x[1], x[2])
)

function main()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 30, 12
    n = 256
    nx = ny = n-1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = range(0, Lx, length=n), range(0, Ly, length=n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xci = xc, yc = range(0+dx/2, Lx-dx/2, length=n-1), range(0+dy/2, Ly-dy/2, length=n-1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv

    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]]);
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]]);
    T  = TA(backend)([y for x in xc, y in yc]);
    V  = Vx, Vy;

    dt = min(dx / maximum(abs.(Array(Vx))),  dy / maximum(abs.(Array(Vy))));
    dt *= 0.5

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1));
    centroid2particle!(pT, xci, T, particles);

    Tmin, Tmax = 0, 1
    @assert all(x -> Tmin ≤ x ≤ Tmax, Array(pT.data[particles.index.data]))

    !isdir("figs") && mkdir("figs")

    niter = 250
    for it in 1:niter
        @show it 
        advection!(particles, RungeKutta2(), V, (grid_vx, grid_vy), dt)
        move_particles!(particles, xvi, particle_args)
        inject_particles!(particles, (pT, ), xci)
            @assert all(x -> Tmin ≤ x ≤ Tmax, Array(pT.data[particles.index.data]))
        particle2centroid!(T, pT, xci, particles)
            @assert all(x -> Tmin ≤ x ≤ Tmax, Array(T))

        if rem(it, 10) == 0
            f, ax, = heatmap(xci..., Array(T), colormap=:batlow)
            streamplot!(ax, g, xvi...)
            save("figs/test_$(it).png", f)
            f
        end
    end

    @show extrema(T)
    println("Finished")
end

main()
