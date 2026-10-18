# Centroid interpolation

Centroid interpolation transfers a field between cell centers and particles:

```text
centroid grid → particles → centroid grid
```

Use `centroid2particle!` to interpolate a centroid field to particles and
`particle2centroid!` to accumulate particle values back to centroids.

## Ghosted centroid arrays

The stored centroid coordinates `particles.xci` include ghost centroids. The
default centroid-to-particle call uses that layout so particles near a domain
boundary still have a complete interpolation stencil:

```julia
using JustPIC

xv = range(0, 1, length=9)
yv = range(0, 1, length=9)
dx = step(xv)
dy = step(yv)
xc = range(dx / 2, 1 - dx / 2, length=8)
yc = range(dy / 2, 1 - dy / 2, length=8)
grid_vx = xv, JustPIC.add_periodic_ghost_nodes(yc)
grid_vy = JustPIC.add_periodic_ghost_nodes(xc), yv
particles = init_particles(JustPIC.CPU, 16, 24, 8, grid_vx, grid_vy)

C = [y for x in particles.xci[1], y in particles.xci[2]]
pC, = init_cell_arrays(particles, Val(1))

centroid2particle!(pC, C, particles)
particle2centroid!(C, pC, particles)
```

`particle2centroid!` accepts `ghost_1`, `ghost_2`, and `ghost_3` to control the
destination layout, just like `particle2grid!`.

## Physical-only centroid arrays

For a centroid field without ghost values, pass `ghosted=false` to
`centroid2particle!`. The coordinate range is then taken from the physical
centroid coordinates:

```julia
C_physical = [y for x in xc, y in yc]

centroid2particle!(pC, C_physical, particles; ghosted=false)

C_out = similar(C_physical)
particle2centroid!(C_out, pC, particles; ghost_1=false, ghost_2=false)
```

Use the ghosted layout for boundary-sensitive calculations unless the physical
only behavior is intentional. Unlike `grid2particle!`, centroid interpolation
uses the `ghosted` keyword rather than per-direction `ghost_i` keywords.
