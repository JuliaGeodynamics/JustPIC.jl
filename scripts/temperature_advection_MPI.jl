using JustPIC

# set_backend("CUDA") # need to restart session if this changes

using CellArrays
using ParallelStencil
using GLMakie
@init_parallel_stencil(CUDA, Float64, 2)

const TA = backend == "CUDA" ? JustPIC.CUDA.CuArray : Array

function init_particles(nxcell, max_xcell, min_xcell, x, y, dx, dy, nx, ny)
    ni = nx, ny
    ncells = nx * ny
    np = max_xcell * ncells
    px, py = ntuple(_ -> @fill(NaN, ni..., celldims=(max_xcell,)) , Val(2))

    inject = @fill(false, nx, ny, eltype=Bool)
    index = @fill(false, ni..., celldims=(max_xcell,), eltype=Bool) 
    
    @parallel_indices (i, j) function fill_coords_index(px, py, index)    
        # lower-left corner of the cell
        x0, y0 = x[i], y[j]
        # fill index array
        for l in 1:nxcell
            @cell px[l, i, j] = x0 + dx * rand(0.05:1e-5:0.95)
            @cell py[l, i, j] = y0 + dy * rand(0.05:1e-5:0.95)
            @cell index[l, i, j] = true
        end
        return nothing
    end

    @parallel (1:nx, 1:ny) fill_coords_index(px, py, index)    

    return Particles(
        (px, py), index, inject, nxcell, max_xcell, min_xcell, np, (nx, ny)
    )
end

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
vx_stream(x, y) =  250 * sin(π*x) * cos(π*y)
vy_stream(x, y) = -250 * cos(π*x) * sin(π*y)
g(x) = Point2f(
    vx_stream(x[1], x[2]),
    vy_stream(x[1], x[2])
)

function main()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 8, 8, 8
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

    particles = init_particles(
        nxcell, max_xcell, min_xcell, xvi..., dxi..., nx, ny
    )

    # Cell fields -------------------------------
    Vx = TA([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
    T  = TA([y for x in xv, y in yv])
    V  = Vx, Vy

    nx_v = (size(T, 1)-2)*dims[1];
    ny_v = (size(T, 2)-2)*dims[2];
    T_v = zeros(nx_v, ny_v)
    T_nohalo = @zeros(size(T).-2)

    dt = min(dx / maximum(abs.(Vx)),  dy / maximum(abs.(Vy)))

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1))
    grid2particle!(pT, xvi, T, particles.coords)
    
    niter = 50
    for iter in 1:niter
        me == 0 && @show iter
        
        # advect particles
        advection_RK!(particles, V, grid_vx, grid_vy, dt, 2 / 3)

        # update halos
        update_cell_halo!(particles.coords..., particle_args...);
        update_cell_halo!(particles.index)
        # shuffle particles
        shuffle_particles!(particles, xvi, particle_args)
        # interpolate T from particle to grid
        particle2grid!(T, pT, xvi, particles.coords)

        @views T_nohalo .= T[2:end-1, 2:end-1]
        gather!(T_nohalo, T_v)

        if me == 0
            x_global = range(0, Lx, length=size(T_v,1))
            y_global = range(0, Ly, length=size(T_v,2))
            f, ax, = heatmap(x_global, y_global, T_v)
            w = 0.504
            offset = 0.5-(w-0.5)
            lines!(ax,
                [0, w, w, 0, 0],
                [0, 0, w, w, 0],
                linewidth = 3
            )
            lines!(ax,
                [0, w, w, 0, 0] .+ offset,
                [0, 0, w, w, 0],
                linewidth = 3
            )
            lines!(ax,
                [0, w, w, 0, 0] .+ offset,
                [0, 0, w, w, 0] .+ offset,
                linewidth = 3
            )
            lines!(ax,
                [0, w, w, 0, 0],
                [0, 0, w, w, 0] .+ offset,
                linewidth = 3
            )
   
            save("figs/T_MPI_$iter.png", f)
        end    

    end
    
    # f, ax, = heatmap(xvi..., T, colormap=:batlow)
    # streamplot!(ax, g, xvi...)
    # f
end

main()
