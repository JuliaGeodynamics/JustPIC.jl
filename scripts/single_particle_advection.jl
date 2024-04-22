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
vx_stream(x, y) =   250 * sin(π*x) * cos(π*y)
vy_stream(x, y) =  -250 * cos(π*x) * sin(π*y)
g(x) = Point2f(
    vx_stream(x[1], x[2]),
    vy_stream(x[1], x[2])
)

function main()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 48, 18
    n = 65
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

    particles = init_particle(
        backend, nxcell, max_xcell, min_xcell, xvi..., dxi..., nx, ny
    )

    # allocate particle field
    particle_args = ()

    # Cell fields -------------------------------
    Vx  = TA([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
    Vy  = TA([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
    V   = Vx, Vy

    dt = min(dx / maximum(abs.(Vx)),  dy / maximum(abs.(Vy)))

    pxv = particles.coords[1].data;
    pyv = particles.coords[2].data;
    idxv = particles.index.data;
    p = [(pxv[idxv][1], pyv[idxv][1])]

    # Advection test
    niter = 250
    for _ in 1:niter
        advection!(particles, RungeKutta2(2/3), V, (grid_vx, grid_vy), dt)
        move_particles!(particles, xvi, particle_args)

        pxv = particles.coords[1].data;
        pyv = particles.coords[2].data;
        idxv = particles.index.data;
        p_i = (pxv[idxv][1], pyv[idxv][1])
        push!(p, p_i)
    end

    f, ax, = streamplot(g, xvi...)
    lines!(ax, p, color=:red)
    scatter!(ax, p[end], color=:black)
    save("single_particle_advection.png", f)
    f

end

main()
