# Vertex interpolation

Vertex interpolation transfers a field between grid vertices and particles:

```text
vertex grid → particles → vertex grid
```

Use `grid2particle!` to interpolate a vertex field to particles and
`particle2grid!` to accumulate particle values back to vertices.

## Ghosted vertex arrays

`init_particles` stores vertex coordinates with one ghost node on each side.
The default calls therefore expect the source and destination arrays to use the
same ghosted layout:

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

T = [y for x in particles.xvi[1], y in particles.xvi[2]]
pT, = init_cell_arrays(particles, Val(1))

grid2particle!(pT, T, particles)
particle2grid!(T, pT, particles)
```

For staggered fields, create one coordinate tuple per component. For example,
the two-dimensional velocity grids are commonly laid out as
`(xv, ghosted_yc)` and `(ghosted_xc, yv)`; each component can be interpolated
with the same two routines.

## Physical-only vertex arrays

If a field has no ghost nodes, disable the corresponding shift. The following
uses an unghosted field in both directions:

```julia
T_physical = [y for x in xv, y in yv]
grid2particle!(pT, T_physical, particles; ghost_1=false, ghost_2=false)

T_out = similar(T_physical)
particle2grid!(T_out, pT, particles; ghost_1=false, ghost_2=false)
```

Set each `ghost_i` keyword independently when only some directions are padded.
The keywords default to `true`.

## PIC/FLIP update

When both the current and previous vertex fields are available,
`grid2particle_flip!` applies a PIC/FLIP blend. `α = 1` is PIC and `α = 0` is
FLIP:

```julia
grid2particle_flip!(pT, T, T_previous, particles; α=0.5)
```

## API

```@docs
grid2particle!
grid2particle_flip!
particle2grid!
```
