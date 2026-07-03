# Validating Advection & Interpolation Changes

Unit tests catch regressions, but PIC changes need *physical* validation: a kernel can pass
tolerance checks while producing drift, clumping or diffusion that only shows up over many
timesteps. Follow this ladder before declaring a change correct.

## 1. Analytic velocity fields

Validate advection against flows with known solutions:

- **Solid-body rotation**: `V = (-y, x)` (2D) — a circular blob must return to its start
  after one period with bounded shape distortion. This is the "Rotating circle" miniapp in
  `test_2D.jl`.
- **Pure shear / cellular flow**: `Vx = εx`, `Vy = -εy` or the sinusoidal cell used in the
  advection miniapps — particle trajectories are integrable by hand.
- Compare RK4 against Euler/RK2: order-of-accuracy should be visible when halving `dt`.

## 2. Conservation & occupancy invariants

After every advection + `move_particles!` (+ injection) cycle, check:

- Total particle count is conserved (or grows only by explicit injection):
  `sum(particles.index.data)`
- No cell drops below `min_xcell` or exceeds `max_xcell` occupancy
- No particle sits outside its owning cell (cell locality restored by `move_particles!`)
- Phase ratios sum to 1 at every center/vertex/velocity node after
  `update_phase_ratios!`

## 3. Interpolation round-trips

- grid → particle → grid of a smooth field must reproduce the field to interpolation
  order; linear fields must be reproduced *exactly* by linear kernels (lerp, LinP)
- Check both `grid2particle!`/`particle2grid!` (vertices) and the centroid variants
- MQS kernels: verify against the plain scheme on a refined grid — differences should be
  small and localized to sharp gradients

## 4. Short runs before long runs

- Run a few hundred timesteps at low resolution on CPU: no NaNs
  (`any(isnan, p.coords[1].data)`), no runaway particle loss, visual sanity of the blob
- Re-run the identical script on GPU: results must match CPU to floating-point
  tolerance. A CPU/GPU mismatch is a bug (usually type instability or a race), not noise
- Marker-chain changes: additionally check the chain stays monotonic in `x` after
  `resample!` and that `compute_rock_fraction!` stays in [0, 1]

## 5. Visual checks

Scatter-plot particle positions colored by phase every N steps (Makie). Clumping, banding,
or empty streaks near cell boundaries indicate injection/move bugs even when tests pass.

## Common Issues

- **NaN blowups**: `dt` too large for the integrator, or backtracking left the domain —
  check the semilagrangian clamp logic before blaming the flow
- **Particle starvation** near inflow boundaries: injection not keeping up; tune
  `min_xcell` or use `force_injection!`
- **CPU/GPU divergence**: type instability, `Float64` literals on Metal, or missing
  `synchronize` before host reads
- **"Nothing happens"**: velocity arrays not on the same backend as the particles —
  `ka_backend` dispatch silently picks the backend of the *particles*, conversions are on
  the caller
