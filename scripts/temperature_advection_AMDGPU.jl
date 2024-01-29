using AMDGPU
const backend = AMDGPUBackend

using JustPIC
using JustPIC._2D
using CairoMakie

function init_particles(nxcell, max_xcell, min_xcell, x, y, dx, dy, nx, ny)
    ni = nx, ny
    ncells = nx * ny
    np = max_xcell * ncells
    px, py = ntuple(_ -> @rand(ni..., celldims=(max_xcell,)) , Val(2))

    inject = @fill(false, nx, ny, eltype=Bool)
    index = @fill(false, ni..., celldims=(max_xcell,), eltype=Bool) 
    
    @parallel_indices (i, j) function fill_coords_index(px, py, index, x, y, dx, dy, nxcell)
        # lower-left corner of the cell
        x0, y0 = x[i], y[j]
        # fill index array
        for l in 1:nxcell
            @cell px[l, i, j] = x0 + dx * (@cell(px[l, i, j]) * 0.9 + 0.05)
            @cell py[l, i, j] = y0 + dy * (@cell(py[l, i, j]) * 0.9 + 0.05)
            @cell index[l, i, j] = true
        end
        return nothing
    end

    @parallel (1:nx, 1:ny) fill_coords_index(px, py, index, x, y, dx, dy, nxcell) 

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

# Analytical flow solution
vx_stream(x, y) =  250 * sin(π*x) * cos(π*y)
vy_stream(x, y) = -250 * cos(π*x) * sin(π*y)
g(x) = Point2f(
    vx_stream(x[1], x[2]),
    vy_stream(x[1], x[2])
)

function main()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 24, 24
    n = 256
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

    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi..., dxi..., nx, ny
    )

    # Cell fields -------------------------------
    Vx = TA([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
    T  = TA([y for x in xv, y in yv])
    V  = Vx, Vy

    dt = min(dx / maximum(abs.(Vx)),  dy / maximum(abs.(Vy)))

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1))
    grid2particle!(pT, xvi, T, particles.coords)
    
    niter = 150
    for _ in 1:niter
        advection_RK!(particles, V, grid_vx, grid_vy, dt, 2 / 3)
        shuffle_particles!(particles, xvi, particle_args)
        particle2grid!(T, pT, xvi, particles.coords)
    end

    f, ax, = heatmap(xvi..., Array(T), colormap=:batlow)
    streamplot!(ax, g, xvi...)
    f
end

f = main()
