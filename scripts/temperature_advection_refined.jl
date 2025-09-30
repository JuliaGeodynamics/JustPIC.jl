using JustPIC
using JustPIC._2D

# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"),
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

using GLMakie

function expand_range(x::AbstractVector)
    dx_left  = x[2] - x[1]
    dx_right = x[end] - x[end-1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx_left
    xF = x2 + dx_right
    return vcat(xI, x, xF)
end

# Analytical flow solution
vx_stream(x, y) = 250 * sin(π * x) * cos(π * y)
vy_stream(x, y) = -250 * cos(π * x) * sin(π * y)
g(x) = Point2f(
    vx_stream(x[1], x[2]),
    vy_stream(x[1], x[2])
)

# checks if grid options are reasonable
function checkGridLength(n, d0, f)
    if f < 1
        error("Growth factor cannot be smaller than 1!")
    elseif isone(f)
        return n*d0
    else
        return d0 * (f^n - 1) / (f - 1)
    end
end

# finds the correct growth factor by bisection
function findGrowthFactor(L, n, d0)
    a = 1.0
    b = 2.0
    for i = 1 : 20
        c = (a+b) / 2.0
        err = checkGridLength(n, d0, c) - L
        if abs(err) < L / 1e3
            println("Grid growth factor: $(c)")
            return c 
        elseif err > 0
            b = c
        else
            a = c
        end
        #println("c: $(c)")
    end
    println("Grid seems impossible!")
end

# make exponential grid
function makeExpoGrid(L, n, d0, x0)
    dx = zeros(n)
    if mod(n,2) == 0
        L2  = L/2.0
        n2  = Int64(n/2)
        f   = findGrowthFactor(L2, n2, d0) 
        dx[n2:n2+1] .= d0
        dn  = 2
    else
        L2 = L/2.0 + d0/2.0
        n2 = Int64((n+1) / 2)
        f  = findGrowthFactor(L2, n2, d0)
        dx[n2] = d0 
        dn = 1
    end
    for i = n2+dn : n-1
        dx[i] = dx[i-1] * f
    end
    for i = n2-1 : -1 : 2
        dx[i] = dx[i+1] * f 
    end

    dx[1]     = (L - sum(dx)) / 2.0
    dx[end]   = dx[1]
    
    xn        = zeros(n+1)
    xc        = zeros(n+2) # with ghost cells
    xn[1]     = x0
    xc[1]     = x0 - dx[1] / 2.0
    xc[end]   = x0 + L + dx[end] / 2.0
    for i = 1 : n
        xn[i+1] = xn[i] + dx[i]
        xc[i+1] = (xn[i] + xn[i+1]) / 2.0
    end

    # dx from the vertices
    return xn, xc[2:end-1], dx
end

function main()

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 30, 12
    n   = 32
    nx  = ny = n - 1
    Lx  = Ly = 1.0
    
    dx0 = Lx / nx
    dy0 = Ly / ny

    # refined coordinates
    xv, xc, dx  = makeExpoGrid(Lx, nx, dx0 / 5, 0e0)
    yv, yc, dy  = makeExpoGrid(Ly, ny, dy0, 0e0)

    xvi = xv, yv
  
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv
    grid_vi = grid_vx, grid_vy

    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
    T  = TA(backend)([y for x in xv, y in yv])
    V  = Vx, Vy

    dt = min(minimum(dx) / maximum(abs.(Array(Vx))), minimum(dy) / maximum(abs.(Array(Vy))))
    dt *= 0.5

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1))
    grid2particle!(pT, xvi, T, particles)

    !isdir("figs") && mkdir("figs")

    f, ax, = heatmap(xvi..., Array(T), colormap = :batlow)
    streamplot!(ax, g, xvi...)
    save("figs/test_0.png", f)
    f

    niter = 50
    # for it in 1:niter
        advection!(particles, RungeKutta2(), V, grid_vi, dt)
        move_particles!(particles, xvi, particle_args)
        # inject_particles!(particles, (pT,), xvi)
        particle2grid!(T, pT, xvi, particles)

    #     if rem(it, 1) == 0
    #         f, ax, = heatmap(xvi..., Array(T), colormap = :batlow)
    #         streamplot!(ax, g, xvi...)
    #         save("figs/test_$(it).png", f)
    #         f
    #     end
    # end

    return println("Finished")
end

main()


# px, py = particles.coords;
# px, py = px.data[:], py.data[:];
# idx = particles.index.data[:];
# scatter(px[idx], py[idx], color = pT.data[:][idx], colormap = :batlow)

# advection_MQS!(particles, RungeKutta2(), V, (grid_vx, grid_vy), dt)
# move_particles!(particles, xvi, particle_args)

# px, py = particles.coords;
# px, py = px.data[:], py.data[:];
# idx = particles.index.data[:];
# scatter(px[idx], py[idx], color = pT.data[:][idx], colormap = :batlow)

# # scatter!(px, py, color="red")
# scatter(px, py, color="red")

import JustPIC._2D as JP

idx       = 16, 1
grid      = xvi
corner_xi = JP.corner_coordinate(grid, idx...)
di        = JP.compute_dx(grid)
dxi       = @dxi di idx...
dxiL      = di[1][idx[1] .- 1]
dxiR      = di[1][idx[1] .+ 1]
px_cell   = particles.coords[1][idx...] 
py_cell   = particles.coords[2][idx...]

ip        = findfirst(px_cell .== 0.5111148788481209)
pᵢ        = px_cell[ip], py_cell[ip]

corner_xi[1] + dxi[1]
corner_xi[1] + dxi[1] + dxiR

dxi[1] / Vx[16, 1]
dxi[2] / Vy[16, 1]

