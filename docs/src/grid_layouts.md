# Grid layouts

JustPIC distinguishes vertex grids, centroid grids, and staggered velocity
grids. In 2D, a staggered velocity field is supplied as:

```julia
grid_vx = xv, yc_ghosted
grid_vy = xc_ghosted, yv
```

The layout can be sketched as follows (`•` are vertices, `×` are centroids;
`u` and `v` are staggered velocity locations):

```text
•──────u──────•──────u──────•
│      ×      │      ×      │
v      │      v      │      v
│      ×      │      ×      │
•──────u──────•──────u──────•
```

The outermost coordinate values used by interpolation are ghost images of the
physical boundary values, not additional physical cells.

The diagonal coordinate of each velocity component is the vertex coordinate;
the off-diagonal coordinate is the cell-center coordinate. `init_particles`
derives `particles.xvi` and `particles.xci` from these grids and adds one
periodic ghost node on each side.

```julia
using JustPIC

xv = range(0, 1, length=9)
yv = range(0, 1, length=9)
dx, dy = step(xv), step(yv)
xc = range(dx / 2, 1 - dx / 2, length=8)
yc = range(dy / 2, 1 - dy / 2, length=8)
grid_vx = xv, add_periodic_ghost_nodes(yc)
grid_vy = add_periodic_ghost_nodes(xc), yv
particles = init_particles(JustPIC.CPU, 16, 24, 8, grid_vx, grid_vy)
```

The resulting layouts are:

| Layout | Coordinates | Typical use |
| --- | --- | --- |
| `particles.xvi` | vertices, ghosted | vertex fields and `grid2particle!` |
| `particles.xci` | centroids, ghosted | centroid fields and `centroid2particle!` |
| `particles.xi_vel` | staggered component grids | velocity advection |

Use `add_periodic_ghost_nodes` for coordinate vectors that need one periodic
image at either boundary. Physical-only fields can still be interpolated by
disabling the relevant `ghost_i` or `ghosted` option; see the interpolation
pages for complete examples.
