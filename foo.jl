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

# RK2 backtracking
function backtrace(xi, dt, vfunc)
    v1 = vfunc(xi)
    x_mid = xi - 0.5 * dt * v1
    v2 = vfunc(x_mid)
    return xi - dt * v2
end

# Analytical flow solution
vx_stream(x, y) = 250 * sin(π * x) * cos(π * y)
vy_stream(x, y) = -250 * cos(π * x) * sin(π * y)
g(x) = Point2f(
vx_stream(x[1], x[2]),
vy_stream(x[1], x[2])
)

# Initialize particles -------------------------------
n = 128
nx = ny = n - 1
Lx = Ly = 1.0
# nodal vertices
xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
# nodal centers
xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1)
# staggered grid velocity nodal locations
grid_vx = xv, expand_range(yc)
grid_vy = expand_range(xc), yv


# Cell fields -------------------------------
Vx = TA(backend)([vx_stream(x, y) for x in grid_vx[1], y in grid_vx[2]])
Vy = TA(backend)([vy_stream(x, y) for x in grid_vy[1], y in grid_vy[2]])
T = TA(backend)([y for x in xv, y in yv])
T0 = TA(backend)([y for x in xv, y in yv])

V = Vx, Vy

dt = min(dx / maximum(abs.(Array(Vx))), dy / maximum(abs.(Array(Vy))))
dt *= 0.75

!isdir("figs") && mkdir("figs")

niter = 250
for it in 1:niter
    # semilagrangian_advection!(T, T0, RungeKutta2(), V, (grid_vx, grid_vy), xvi, dt)
    semilagrangian_advection_LinP!(T, T0, RungeKutta2(), V, (grid_vx, grid_vy), xvi, dt)
    # semilagrangian_advection_MQS!(T, T0, RungeKutta2(), V, (grid_vx, grid_vy), xvi, dt)
    T[1,:]    .= T[2,:]
    T[end,:]  .= T[end-1,:] 
    # T[:, 1]   .= T[:, 2]
    # T[:, end] .= T[:, end-1] 
    copyto!(T0, T)

    if rem(it, 1) == 0
        f, ax, = heatmap(xvi..., Array(T), colormap = :batlow)
        # streamplot!(ax, g, xvi...)
        save("figs/test_$(it).png", f)
        f
    end
end
