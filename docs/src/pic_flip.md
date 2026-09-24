# PIC and FLIP

`grid2particle_flip!` combines particle values with grid updates. It is useful
when pure PIC is too diffusive but pure FLIP is too noisy.

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
