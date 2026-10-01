# PIC and FLIP

FLIP transfers combine particle values with grid updates. They are useful when
pure PIC is too diffusive but pure FLIP is too noisy. JustPIC provides them in
both directions and for both vertex and centroid grids:

| Direction | Vertices | Centroids |
| --- | --- | --- |
| grid → particles (blend) | [`grid2particle_flip!`](@ref) | [`centroid2particle_flip!`](@ref) |
| particles → grid (increment) | [`particle2grid_flip!`](@ref) | [`particle2centroid_flip!`](@ref) |

All of them accept single fields or tuples of fields in 2D and 3D, and run
unchanged on every backend, including `Float32`.

## Grid to particles

Given current grid field `F`, previous grid field `F₀`, and the existing particle
field `Fp`:

```julia
grid2particle_flip!(Fp, F, F₀, particles; α=0.5)
```

The blend parameter is:

| `α` | Method |
| ---: | --- |
| `1` | pure PIC: replace with interpolated current-grid value |
| `0` | pure FLIP: add the interpolated grid change to the particle value |
| between `0` and `1` | PIC/FLIP blend |

Conceptually, the blend combines the current-grid estimate with the particle
value plus the grid increment:

```text
current grid ──interpolate──▶ PIC value ──┐
                                          ├── α PIC + (1 − α) FLIP ──▶ particle
old/current grids ──difference──▶ Δgrid ──┘
particle + Δgrid ───────────────────────▶ FLIP value
```

The usual time-step pattern is:

```julia
F₀ .= F
# update F on the grid
grid2particle_flip!(Fp, F, F₀, particles; α=0.5)
```

The source arrays use the same ghost layout rules as `grid2particle!`; pass
`ghost_1`, `ghost_2`, and `ghost_3` when a direction is physical-only.

`centroid2particle_flip!(Fp, F, F₀, particles; α)` does the same with
cell-centered fields in the `particles.xci` layout; its `ghosted`, `ghost_1`,
`ghost_2` and `ghost_3` keywords are those of [`centroid2particle!`](@ref).

## Particles to grid

The reverse transfer accumulates the change of a particle field, `Fp - Fp₀`,
onto a grid field with the same weights as [`particle2grid!`](@ref) and
[`particle2centroid!`](@ref):

```julia
particle2grid_flip!(F, Fp, Fp₀, particles)      # F += Σ ω (Fp - Fp₀) / Σ ω
particle2centroid_flip!(Fc, Fp, Fp₀, particles)
```

`F` is only modified where at least one particle contributes (`Σ ω ≠ 0`), so
the result carries over the previous grid value elsewhere. No temporary arrays
or atomics are involved: each grid node gathers from the particles around it.

## API

```@docs
grid2particle_flip!
centroid2particle_flip!
particle2grid_flip!
particle2centroid_flip!
```
