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

# Background flow (3D analytical solenoidal field)
Vx = TA(backend)([0.1 * sin(π * x) * cos(π * z) for x in grid_vxi[1][1], y in grid_vxi[1][2], z in grid_vxi[1][3]])
Vy = TA(backend)([0.0 for x in grid_vxi[2][1], y in grid_vxi[2][2], z in grid_vxi[2][3]])
Vz = TA(backend)([-0.1 * cos(π * x) * sin(π * z) for x in grid_vxi[3][1], y in grid_vxi[3][2], z in grid_vxi[3][3]])
V = Vx, Vy, Vz

# Initialize the surface with a small bump
z_init = [
    0.5 + 0.05 * exp(-50 * ((x - 0.1)^2 + (y - 0.5)^2))
        for x in xv, y in yv
]
surf = init_marker_surface(backend, xv, yv, z_init; periodic_1 = true)

# Time stepping
dt = 0.05
for _ in 1:25
    advect_marker_surface!(surf, V, grid_vxi, dt; max_slope_angle = 45.0)
    f = Figure()
    ax = Axis3(f[1, 1]; aspect = (1, 1, 0.5))
    surface!(ax, xv, yv, Array(surf.topo); colormap = :terrain)
    display(f)
    sleep(0.5)
end

# Plot the deformed surface
f = Figure()
ax = Axis3(f[1, 1]; aspect = (1, 1, 0.5))
surface!(ax, xv, yv, Array(surf.topo); colormap = :terrain)
display(f)
