ENV["PS_PACKAGE"] = "Threads"

using JustPIC
using CellArrays
using GLMakie
using ImplicitGlobalGrid
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

const TA = ENV["PS_PACKAGE"] == "CUDA" ? JustPIC.CUDA.CuArray : Array

function init_particle(nxcell, max_xcell, min_xcell, x, y, dx, dy, nx, ny)
    ni = nx, ny
    ncells = nx * ny
    np = max_xcell * ncells
    px, py = ntuple(_ -> @fill(NaN, ni..., celldims=(max_xcell,)) , Val(2))

    inject = @fill(false, nx, ny, eltype=Bool)
    index = @fill(false, ni..., celldims=(max_xcell,), eltype=Bool) 

    i  ,j  = Int(nx÷3), Int(ny÷3)
    x0, y0 = x[i], y[j]

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

function expand_range(x::AbstractArray, dx)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    x = TA(vcat(xI, x, xF))
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
    n   = 64
    nx  = ny = n-1
    igg = init_global_grid(n, n, 0; init_MPI=false)
    Lx  = Ly = 1.0
    dxi = dx, dy = Lx /nx_g(), Ly / ny_g()
    # nodal vertices
    xv  = @zeros(n) 
    yv  = @zeros(n) 
    xv .= TA([x_g(i, dx, xv) for i in eachindex(xv)])
    yv  = deepcopy(xv)
    xvi = xv, yv
    # nodal centers
    xc  = @zeros(nx) 
    xc .= TA([x_g(i, dx, xc) for i in eachindex(xc)])
    yc  = deepcopy(xc)
    xci = xc, yc
    # staggered grid for the velocity components
    grid_vx = xv, expand_range(yc, dy)
    grid_vy = expand_range(xc, dx), yv

    particles = init_particle(
        nxcell, max_xcell, min_xcell, xv, yv, dxi..., nx, ny
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
    niter = 150
    for iter in 1:niter
        advection_RK!(particles, V, grid_vx, grid_vy, dt, 2 / 3)
        shuffle_particles!(particles, xvi, particle_args)

        update_halo!(particles.coords[1].data);
        update_halo!(particles.coords[2].data);
        update_halo!(particles.index.data);
        for arg in particle_args
            update_halo!(arg)
        end

        pxv = particles.coords[1].data;
        pyv = particles.coords[2].data;
        idxv = particles.index.data;
        p_i = (pxv[idxv][1], pyv[idxv][1])
        # @show p_i, iter
        push!(p, p_i)
    end

    f, ax, = streamplot(g, xvi...)
    lines!(ax, p, color=:red)
    f

end

@time f = main()
# save("single_particle_advection.png", f)
# f