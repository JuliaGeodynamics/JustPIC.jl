using JustPIC

# set_backend("CUDA") # need to restart session if this changes

using CellArrays
using ParallelStencil
using GLMakie
@init_parallel_stencil(CUDA, Float64, 2)

const TA = backend == "CUDA" ? JustPIC.CUDA.CuArray : Array

function init_particle(nxcell, max_xcell, min_xcell, x, y, dx, dy, nx, ny)
    ni = nx, ny
    ncells = nx * ny
    np = max_xcell * ncells
    px, py = ntuple(_ -> @fill(NaN, ni..., celldims=(max_xcell,)) , Val(2))

    inject = @fill(false, nx, ny, eltype=Bool)
    index = @fill(false, ni..., celldims=(max_xcell,), eltype=Bool) 

    i , j  = Int(nx÷3), Int(ny÷3)
    x0, y0 = x[i], y[j]
    @show i,j
    @cell    px[1, i, j] = x0 + dx * rand(0.05:1e-5: 0.95)
    @cell    py[1, i, j] = y0 + dy * rand(0.05:1e-5: 0.95)
    @cell index[1, i, j] = true

    return Particles(
        (px, py), index, inject, nxcell, max_xcell, min_xcell, np, (nx, ny)
    )
end

@inline init_particle_fields_cellarrays(particles, ::Val{N}) where N = ntuple(_ -> @fill(0.0, size(particles.coords[1])..., celldims=(cellsize(particles.index))), Val(N))

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
        nxcell, max_xcell, min_xcell, xvi..., dxi..., nx, ny
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
    for iter in 1:niter
        advection_RK!(particles, V, grid_vx, grid_vy, dt, 2 / 3)
        shuffle_particles!(particles, xvi, particle_args)

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

@time f = main()
