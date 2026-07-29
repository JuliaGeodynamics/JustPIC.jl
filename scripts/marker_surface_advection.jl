using JustPIC
using GLMakie

const backend = JustPIC.CPU

# Initialize domain & grids
n = 64
Lx = Ly = Lz = 1.0
xv = LinRange(0, Lx, n)
yv = LinRange(0, Ly, n)
zv = LinRange(0, Lz, n)
xvi = (collect(xv), collect(yv), collect(zv))

# Background flow (3D analytical solenoidal field)
Vx = TA(backend)([ 0.10 * sin(π*x) * cos(π*z)             for x in xv, y in yv, z in zv])
Vy = TA(backend)([ 0.0                                    for x in xv, y in yv, z in zv])
Vz = TA(backend)([-0.10 * cos(π*x) * sin(π*z)             for x in xv, y in yv, z in zv])
V  = Vx, Vy, Vz

# Initialize the surface with a small bump
z_init = [0.5 + 0.05 * exp(-50 * ((x-0.1)^2 + (y-0.5)^2))
          for x in xv, y in yv]
surf = init_marker_surface(backend, xv, yv, z_init; periodic_1=true)

# Time stepping
dt = 0.05
for _ in 1:25
    advect_marker_surface!(surf, V, xvi, dt; max_slope_angle = 45.0)
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
