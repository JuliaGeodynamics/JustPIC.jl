ENV["PS_PACKAGE"] = "Threads"

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

using JustPIC
using JustPIC.ImplicitGlobalGrid
using CellArrays
using GLMakie
# using ImplicitGlobalGrid
import MPI

const TA = ENV["PS_PACKAGE"] == "CUDA" ? JustPIC.CUDA.CuArray : Array

function init_particle(nxcell, max_xcell, min_xcell, x, y, dx, dy, nx, ny)
    ni = nx, ny
    ncells = nx * ny
    np = max_xcell * ncells
    px, py = ntuple(_ -> @fill(NaN, ni..., celldims=(max_xcell,)) , Val(2))

    inject = @fill(false, nx, ny, eltype=Bool)
    index = @fill(false, ni..., celldims=(max_xcell,), eltype=Bool) 

    i  ,j  = Int(nx_g()÷3), Int(nx_g()÷3)
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
    nxcell, max_xcell, min_xcell = 24, 24, 18
    n   = 32
    nx  = ny = n-1
    me, dims, = init_global_grid(n-1, n-1, 0; init_MPI=JustPIC.MPI.Initialized() ? false : true)
    Lx  = Ly = 1.0
    dxi = dx, dy = Lx /(nx_g()-1), Ly / (ny_g()-1)
    # nodal vertices
    xvi = xv, yv = let
        dummy = zeros(n, n) 
        xv  = TA([x_g(i, dx, dummy) for i in axes(dummy, 1)])
        yv  = TA([y_g(i, dx, dummy) for i in axes(dummy, 2)])
        xv, yv
    end
    # nodal centers
    xci = xc, yc = let
        dummy = zeros(nx, ny) 
        xc  = @zeros(nx) 
        xc .= TA([x_g(i, dx, dummy) for i in axes(dummy, 1)])
        yc  = TA([y_g(i, dx, dummy) for i in axes(dummy, 2)])
        xc, yc
    end

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

    # nx_v = (size(Vx, 1)-2)*dims[1];
    # ny_v = (size(Vx, 2)-2)*dims[2];
    # Vx_v = zeros(nx_v, ny_v)
    # Vx_nohalo = @zeros(size(Vx).-2)

    # nx_v = (size(Vy, 1)-2)*dims[1];
    # ny_v = (size(Vy, 2)-2)*dims[2];
    # Vy_v = zeros(nx_v, ny_v)
    # Vy_nohalo = zeros(size(Vy).-2)

    # @views Vx_nohalo .= Vx[2:end-1, 2:end-1]
    # gather!(Vx_nohalo, Vx_v)
    # @views Vy_nohalo .= Vy[2:end-1, 2:end-1]
    # gather!(Vy_nohalo, Vy_v)

    # if me == 0
    #     f = heatmap(Vx_v)
    #     save("Vx_MPI.png", f)
    #     f = heatmap(Vy_v)
    #     save("Vy_MPI.png", f)
    # end

    dt = min(dx / maximum(abs.(Vx)),  dy / maximum(abs.(Vy)))

    nx_v = (size(particles.coords[1].data, 2))*dims[1];
    ny_v = (size(particles.coords[1].data, 3)-2)*dims[2];
    px_v = fill(NaN, nx_v, ny_v)
    py_v = fill(NaN, nx_v, ny_v)
    index_v = fill(false, nx_v, ny_v)
    px_nohalo = fill(NaN, size(particles.coords[1].data, 2), size(particles.coords[1].data, 3)-2)
    py_nohalo = fill(NaN, size(particles.coords[1].data, 2), size(particles.coords[1].data, 3)-2)
    index_nohalo = fill(false, size(particles.coords[1].data, 2), size(particles.coords[1].data, 3)-2)

    p = [(NaN, NaN)]

    # Advection test
    niter = 50
    for iter in 1:niter
        @show iter
        
        advection_RK!(particles, V, grid_vx, grid_vy, dt, 2 / 3)
        shuffle_particles!(particles, xvi, particle_args)

        update_cell_halo!(particles.coords[1]);
        update_cell_halo!(particles.coords[2]);
        update_cell_halo!(particles.index);
        for arg in particle_args
            update_cell_halo!(arg)
        end

        @views px_nohalo .= particles.coords[1].data[1, :, 2:end-1]
        @views py_nohalo .= particles.coords[2].data[1, :, 2:end-1]
        @views index_nohalo .= particles.index.data[1, :, 2:end-1]
        gather!(px_nohalo, px_v)
        gather!(py_nohalo, py_v)
        gather!(index_nohalo, index_v)

        if me == 0
            p_i = (px_v[index_v][1], py_v[index_v][1])
            push!(p, p_i)
            w = 0.504
            f, ax, = lines(
                [0, w, w, 0, 0],
                [0, 0, w, w, 0],
                linewidth = 3
            )
            lines!(ax,
                [0, w, w, 0, 0].+w,
                [0, 0, w, w, 0],
                linewidth = 3
            )
            lines!(ax,
                [0, w, w, 0, 0].+w,
                [0, 0, w, w, 0].+w,
                linewidth = 3
            )
            lines!(ax,
                [0, w, w, 0, 0],
                [0, 0, w, w, 0].+w,
                linewidth = 3
            )
            streamplot!(ax, g, LinRange(0, 1, 100), LinRange(0, 1, 100))
            lines!(ax, p, color=:red)
            save("figs/trajectory_MPI_$iter.png", f)
        end
    end

    finalize_global_grid();
    # if me == 0
    #     f, ax, = streamplot(g, LinRange(0, 1, 100), LinRange(0, 1, 100))
    #     lines!(ax, p, color=:red)
    #     save("trajectory_MPI.png", f)
    # end
end

main()