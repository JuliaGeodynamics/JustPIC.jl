# Periodic donut advection in 2D

The script
[`scripts/donut_advection_periodic.jl`](https://github.com/JuliaGeodynamics/JustPIC.jl/blob/main/scripts/donut_advection_periodic.jl)
shows how to advect a compact scalar structure through a rectangular domain
with periodic boundary conditions in the horizontal direction.

The setup uses a constant analytical velocity field:

```julia
vx_stream(x, y) = 1.0
vy_stream(x, y) = 0.0
```

Particles are initialized on the staggered velocity grid in the usual way:

```julia
grid_vx = xv, expand_range(yc)
grid_vy = expand_range(xc), yv
particles = init_particles(
    backend, nxcell, max_xcell, min_xcell, grid_vx, grid_vy
)
```

The scalar field is initialized as an annulus, which makes it easy to inspect
how well the particle-to-grid transfer preserves a sharp structure during
transport:

```julia
@inline incircles(x, y, xc, yc, r1, r2) =
    r1^2 ≤ (x - xc)^2 + (y - yc)^2 ≤ r2^2

T = TA(backend)([
    incircles(x, y, xc, yc, r1, r2) * 1.0
    for x in particles.xvi[1], y in particles.xvi[2]
])
```

After interpolating `T` to the particle field with `grid2particle!`, the time
loop advances particles with `advection!`, restores cell sorting with
`move_particles!`, injects particles where needed, and reconstructs the field
on the grid:

```julia
for _ in 1:frame_stride
    advection!(particles, RungeKutta2(), V, dt)
    move_particles!(particles, particle_args; periodic_1 = true, periodic_2 = false)
    inject_particles!(particles, (pT,))
    particle2grid!(T, pT, particles)
end
```

The key detail is the call to `move_particles!` with `periodic_1 = true`. This
wraps particles that leave the domain in the first coordinate direction back to
the opposite side, while the second direction remains non-periodic.

The script also uses `GLMakie` to record a GIF of the evolving annulus, making
it a convenient example for checking periodic transport and visualization
together.

In practice, this example is most useful when you want to verify three things
at once:

- particles can leave and re-enter the domain cleanly in the periodic direction,
- `inject_particles!` maintains particle density during transport,
- the reconstructed annulus stays coherent after repeated `particle2grid!`
  transfers.
