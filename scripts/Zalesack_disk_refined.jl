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
vi_stream(x) = π * 1.0e-5 * (x - 0.5)
g(x) = Point2f(
    -vi_stream(x[2]),
    vi_stream(x[1])
)

"""
    in_zalesak_disk(x, y; xc=0.5, yc=0.75, R=0.15, slot_width=0.05, slot_depth=0.25)

Check if the point (x, y) lies inside the Zalesak disk.

The Zalesak disk is a circle of radius `R` centered at (`xc`, `yc`),
with a vertical rectangular slot of width `slot_width` and depth `slot_depth`
cut out from the disk (the slot extends downward from the disk center).
"""
function in_zalesak_disk(x, y; xc=0.5, yc=0.75, R=0.15, slot_width=0.05, slot_depth=0.25)
    # Check if point is inside the circular disk
    inside_circle = (x - xc)^2 + (y - yc)^2 <= R^2

    # Define the slot region (a vertical rectangle centered on xc)
    slot_left   = xc - slot_width / 2
    slot_right  = xc + slot_width / 2
    slot_bottom = yc - R
    slot_top    = yc - R + slot_depth

    inside_slot = (slot_left <= x <= slot_right) && (slot_bottom <= y <= slot_top)

    # Point is inside the disk but not in the slot
    return inside_circle && !inside_slot
end

function main()
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 25, 35, 10
    n = 101
    nx = ny = n - 1
    Lx = Ly = 1.0

    dx0 = Lx / nx
    dy0 = Ly / ny
    
    # refined coordinates
    xv, xc, dx  = makeExpoGrid(Lx, nx, dx0 / 2, 0e0)
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
        backend, nxcell, max_xcell, min_xcell, xvi_device...,
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([-vi_stream(y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([ vi_stream(x) for x in grid_vy[1], y in grid_vy[2]])
    T = TA(backend)([ in_zalesak_disk(x, y; R=0.1, slot_width=0.05/2, slot_depth=0.15) * 1.0 for x in xv, y in yv])
    
    V = Vx, Vy

    w = π * 1.0e-5  # angular velocity
    period = 1  # revolution number
    tmax = period / (w / (2 * π))
    dt = 200.0

    particle_args = pT, = init_cell_arrays(particles, Val(1))
    grid2particle!(pT, xvi, T, particles)

    t      = 0
    it     = 0
    t_pic  = 0.0
    f, ax, h = heatmap(xvi..., Array(T), colormap = :batlow, colorrange=(0,1))
    streamplot!(ax, g, xvi...)
    Colorbar(f[1,2], h)
    save("figs_refined/test_$(it).png", f)
    println("Saved figure at t = $t s")
    
    local f
    while t ≤ tmax
        advection!(particles, RungeKutta2(), V, grid_vi_device, dt)
        move_particles!(particles, xvi_device, particle_args)
        
        inject_particles!(particles, (pT,), xvi_device)
        particle2grid!(T, pT, xvi_device, particles)

        t += dt
        it += 1
        if isone(it) || rem(it, 50) == 0
            f, ax, h = heatmap(xvi..., Array(T), colormap = :batlow, colorrange=(0,1))
            streamplot!(ax, g, xvi...)
            Colorbar(f[1,2], h)
            save("figs_refined/test_$(it).png", f)
            println("Saved figure at t = $t s")
        end
    end
    # display(f)

    return println("Finished, with t_pic = $t_pic s")
end

main()
