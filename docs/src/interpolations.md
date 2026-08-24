# Interpolations

The interpolation routines in JustPIC target rectangular 2D and 3D grids and
support both uniform and coordinate-vector refined grids.

## Grid to particle

Information is transferred from nodal or staggered grids to particle-carried
fields with linear interpolation. The one-dimensional linear interpolation
kernel is

$v_{\text{p}} = t v_0  + (1 -t) v_1$

where the $t$, $v_0$, and $v_1$ are graphically described below.

<img src="assets/lerp.png" width="250"  />

Numerically, it is more appropriately implemented as a double [fma](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation) as it is slightly more accurate than a naive implementation:

```julia
v_p = fma(t, v1, fma(-t, v0, v0))
```

Bi- and tri-linear interpolation over a rectangular or cubic cells is thus nothing else than a chain of lerp kernels along the different dimensions of the cell. For example, the bilinear interpolation requires two lerps along the left and right sides of the cell, followed by a lerp on the horizontal direction; and trilinear interpolation requires two bilinear kernels and one lerp.

N-linear interpolation is implemented recursively to keep the code dimension
agnostic while staying friendly to compiler specialization.

We can interpolate an arbitrary field `F` onto particles with `grid2particle!`:

```julia
using JustPIC
# define model domain
nxcell, max_xcell, min_xcell = 24, 30, 12
n = 129
Lx  = Ly = 1.0
xvi = xv, yv = range(0, Lx, length=n), range(0, Ly, length=n)
dx, dy = step(xv), step(yv)
xc = range(dx / 2, Lx - dx / 2, length=n - 1)
yc = range(dy / 2, Ly - dy / 2, length=n - 1)
grid_vx = xv, range(first(yc) - dy, step=dy, length=n + 1)
grid_vy = range(first(xc) - dx, step=dx, length=n + 1), yv
particles = init_particles(
    JustPIC.CPU, nxcell, max_xcell, min_xcell, grid_vx, grid_vy
)
# field F at the grid
F = [y for x in particles.xvi[1], y in particles.xvi[2]]
# instantiate empty `CellArray`
Fp, = init_cell_arrays(particles, Val(1));
# interpolate F onto Fp
grid2particle!(Fp, F, particles);
```

`particles.xvi` includes one ghost node on each side, so the compact call above
expects `F` to use that same layout. For a field stored only on physical nodes,
disable the shift in each unpadded direction:

```julia
F_physical = [y for x in xv, y in yv]
grid2particle!(Fp, F_physical, particles; ghost_1=false, ghost_2=false)
```

## Particle to grid

Information on particles can be accumulated back to grid nodes with inverse
distance weighting:

$v_{i,j} = \frac{\sum^N_{k=1} \omega_k v_k}{\sum^n_{k=1} \omega_k}$

where the weight is $\omega_i = d^{-n}$, with $d$ being the distance between the particle and the node, and $n$ a integer number.

On shared-memory hardware this typically requires atomics. JustPIC avoids that
by looping over grid nodes and scanning only the neighboring particle cells that
can contribute to each node.

This interpolation is handled by `particle2grid!`:
```julia-repl
julia> particle2grid!(F, Fp, particles)
```

The same `ghost_1`, `ghost_2`, and `ghost_3` keywords select whether each
destination direction includes particle ghost nodes. They default to `true`.

`particle2centroid!` and `grid2particle_flip!` take the same keywords; for
`particle2centroid!` they refer to the ghosted centroid grid `particles.xci`.
`centroid2particle!` has no opt-out: particles sitting between a domain boundary
and the first centroid are interpolated from the ghost centroids, so its source
field must always use the `particles.xci` layout.

Related high-level helpers in this workflow are `particle2centroid!`,
`centroid2particle!`, `update_phase_ratios!`, `subgrid_diffusion!`, and
`subgrid_diffusion_centroid!`. The two subgrid-diffusion routines read their
`T_grid` through `grid2particle!`/`centroid2particle!` and so expect the ghosted
vertex and centroid layouts respectively, while `ΔT_grid` carries one ghost node
per side, matching `size(particles.index)`.
