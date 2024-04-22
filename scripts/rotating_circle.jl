using JustPIC
using JustPIC._2D
# Threads is the default backend, 
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"), 
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

using GLMakie

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    range(xI, xF, length=n+2)
end

# Analytical flow solution
vi_stream(x) =  π*1e-5 * (x - 0.5)
g(x) = Point2f(
    -vi_stream(x[2]),
     vi_stream(x[1])
)

function main()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 25, 35, 10
    n = 201
    nx = ny = n-1
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = range(0, Lx, length=n), range(0, Ly, length=n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xc, yc = range(0+dx/2, Lx-dx/2, length=n-1), range(0+dy/2, Ly-dy/2, length=n-1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv
    grid_vxi = grid_vx, grid_vy

    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi..., dxi..., nx, ny
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([-vi_stream(y) for x in grid_vx[1], y in grid_vx[2]]);
    Vy = TA(backend)([ vi_stream(x) for x in grid_vy[1], y in grid_vy[2]]);

    xc0 = yc0 =  0.25
    R   = 20 * dx
    T   = TA(backend)([ ((x-xc0)^2 + (y-yc0)^2 ≤ R^2)  * 1.0 for x in xv, y in yv]);
    V   = Vx, Vy;

    w      = π * 1e-5  # angular velocity
    period = 1  # revolution number
    tmax   = period / (w/(2*π))
    dt     = 200.0

    particle_args = pT, = init_cell_arrays(particles, Val(1));
    grid2particle!(pT, xvi, T, particles);
    
    t   = 0
    it  = 0
    t_pic = 0.0
    # inject_particles!(particles, (pT, ), xvi)
    while t ≤ tmax
            advection!(particles, RungeKutta2(), V, grid_vxi, dt)
            move_particles!(particles, xvi, particle_args)
            inject_particles!(particles, (pT, ), xvi)
            particle2grid!(T, pT, xvi, particles)

        t += dt
        it += 1
        if rem(it, 10) == 0
            f, ax, = heatmap(xvi..., Array(T), colormap=:batlow)
            streamplot!(ax, g, xvi...)
            display(f)
            # save("figs/test_$(it).png", f)
        end
    end

    println("Finished, with t_pic = $t_pic s")
end

main()