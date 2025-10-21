using JustPIC
using JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA"),
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU")
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

using GLMakie

function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return LinRange(xI, xF, n + 2)
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
    # nodal vertices
   # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1)
      # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv
    grid_vxi = grid_vx, grid_vy

    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...
    )

    # Cell fields -------------------------------
    Vx = TA(backend)([-vi_stream(y) for x in grid_vx[1], y in grid_vx[2]])
    Vy = TA(backend)([ vi_stream(x) for x in grid_vy[1], y in grid_vy[2]])

    xc0 = yc0 = 0.25
    R = 20 * dx
    T = TA(backend)([ in_zalesak_disk(x, y;R=0.1, slot_width=0.05/2, slot_depth=0.15) * 1.0 for x in xv, y in yv])
    T0 = deepcopy(T)
    
    V = Vx, Vy

    w = π * 1.0e-5  # angular velocity
    # w = 1 # angular velocity
    period = 1  # revolution number
    tmax = period / (w / (2 * π))
    # dt = 2π / 30
    dt = 200.0

    particle_args = pT, = init_cell_arrays(particles, Val(1))
    grid2particle!(pT, xvi, T, particles)

    t = 0
    it = 0
    t_pic = 0.0
    # inject_particles!(particles, (pT, ), xvi)
    local f
    while t ≤ tmax
        advection!(particles, RungeKutta2(), V, grid_vxi, dt)
        move_particles!(particles, xvi, particle_args)
        inject_particles!(particles, (pT,), xvi)
        particle2grid!(T, pT, xvi, particles)

        # semilagrangian_advection!(T, T0, RungeKutta2(), V, (grid_vx, grid_vy), xvi, dt)
        # # T[1,:]    .= T[2,:]
        # # T[end,:]  .= T[end-1,:]
        # # T[:, 1] .= T[:, 2]
        # # T[:, end] .= T[:, end - 1]
        # copyto!(T0, T)

        t += dt
        it += 1
        if rem(it, 100) == 0
            f, ax, = heatmap(xvi..., Array(T), colormap = :batlow)
            streamplot!(ax, g, xvi...)
            save("figs/test_$(it).png", f)
            println("Saved figure at t = $t s")
        end
    end
    # display(f)

    return println("Finished, with t_pic = $t_pic s")
end

main()
