using JustPIC
using GLMakie

const backend = JustPIC.CPU

# Initialize domain & grids
n = 64
Lx = Ly = Lz = 1.0
xv = LinRange(0, Lx, n)
yv = LinRange(0, Ly, n)
zv = LinRange(0, Lz, n)
extend_centers(x) = range(first(x) - step(x) / 2; step = step(x), length = length(x) + 1)

grid_vxi = (
    (xv, extend_centers(yv), extend_centers(zv)),
    (extend_centers(xv), yv, extend_centers(zv)),
    (extend_centers(xv), extend_centers(yv), zv),
)

# Uniform advection in the x direction
Vx = TA(backend)(fill(0.2, length.(grid_vxi[1])))
Vy = TA(backend)(fill(0.0, length.(grid_vxi[2])))
Vz = TA(backend)(fill(0.0, length.(grid_vxi[3])))
V = Vx, Vy, Vz

# Bump placed near the right boundary, so it crosses the periodic seam early on
wrapped_distance(x, x0, L) = min(abs(x - x0), L - abs(x - x0))
z_init = [
    0.5 + 0.05 * exp(-50 * (wrapped_distance(x, 0.9, Lx)^2 + (y - Ly / 2)^2))
        for x in xv, y in yv
]
surf = init_marker_surface(backend, xv, yv, z_init; periodic_1 = true)

# Time stepping
dt = 0.05
for _ in 1:25
    advect_marker_surface!(surf, V, grid_vxi, dt; max_slope_angle = 45.0)
    f = Figure()
    ax = Axis3(f[1, 1]; aspect = (1, 1, 0.5))
    surface!(ax, xv, yv, Array(surf.topo); colormap = :oleron)
    display(f)
    sleep(0.5)
end

# Plot the deformed surface
f = Figure()
ax = Axis3(f[1, 1]; aspect = (1, 1, 0.5))
surface!(ax, xv, yv, Array(surf.topo); colormap = :oleron)
display(f)
