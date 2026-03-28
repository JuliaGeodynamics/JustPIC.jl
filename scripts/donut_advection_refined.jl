# using CUDA
using JustPIC
using JustPIC._2D

# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"),
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
# const backend = CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
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
            # println("Grid growth factor: $(c)")
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


# Analytical flow solution
vx_stream(x, y) = 1.0
vy_stream(x, y) = 0.0
g(x) = Point2f(
    vx_stream(x[1], x[2]),
    vy_stream(x[1], x[2])
)

@inline incircles(x,y, xc, yc, r1, r2) = r1^2 ≤ (x - xc)^2 + (y - yc)^2 ≤ r2^2

function main()
   # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 25, 35, 10
    n = 101
    nx = ny = n - 1
    Lx = 5e0
    Ly = 1.0
   
    dx0 = Lx / nx
    dy0 = Ly / ny
    
    # refined coordinates
    xv, xc, dx  = makeExpoGrid(Lx, nx, dx0 / 1, 0e0)
    yv, yc, dy  = makeExpoGrid(Ly, ny, dy0 / 1, 0e0)

    xvi = xv, yv
  
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv
    grid_vi = grid_vx, grid_vy

    xvi_device = TA(backend).(xvi)
    grid_vi_device = (
        TA(backend).(grid_vi[1]),
        TA(backend).(grid_vi[2]),
    )

    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...,
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
    xc, yc, r1, r2 = 1.0, 0.5, 0.1, 0.3
    T = TA(backend)([incircles(x,y, xc, yc, r1, r2) * 1e0 for x in xv, y in yv])
    V = Vx, Vy

    # dt = min(dx / maximum(abs.(Array(Vx))), dy / maximum(abs.(Array(Vy))))
    # dt *= 0.75

    dt = 0.018 / 2

    # Advection test
    particle_args = pT, = init_cell_arrays(particles, Val(1))
    grid2particle!(pT, xvi, T, particles)

    fname = "figs_$dt"
    !isdir(fname) && mkdir(fname)

    # niter = 120
    t = 0
    it = 0
    # for it in 1:niter
    while t < 3
        it += 1 
        advection!(particles, RungeKutta2(), V, grid_vi_device, dt)
        move_particles!(particles, xvi_device, particle_args)
        inject_particles!(particles, (pT,), xvi_device)
        particle2grid!(T, pT, xvi_device, particles)

        t += dt
        if rem(it, 10) == 0
            f = Figure(size = (800, 160)) 
            ax = Axis(f[1, 1], aspect=5, title = "Donut Advection - Δt = $dt", xlabel = "x", ylabel = "y")   
            heatmap!(ax, xvi..., Array(T), colormap = :batlow)
            # streamplot!(ax, g, xvi...)
            save(joinpath(fname, "test_$(it).png"), f)
            f
        end
    end

    return println("Finished")
end

main()
