# Public API

The public API is organized by workflow rather than as one long flat symbol
list. Use this page as a map and the linked sections for method-level details.

## Main Entry Points

- Particle containers and initialization: see [Particles](particles.md)
- Grid/particle transfers and interpolation schemes: see [Interpolations](interpolations.md)
- Velocity interpolation variants: see [Velocity Interpolation](velocity_interpolation.md)
- Marker-chain surface tracking: see [Marker chain](marker_chain.md)
- Checkpointing and restart I/O: see [I/O](IO.md)

## Core Names

The most commonly used exported names are:

- Particle containers: `Particles`, `PassiveMarkers`, `MarkerChain`
- Particle initialization and management: `init_particles`, `init_passive_markers`, `move_particles!`, `inject_particles!`, `inject_particles_phase!`, `clean_particles!`, `force_injection!`
- Interpolation: `grid2particle!`, `grid2particle_flip!`, `particle2grid!`, `centroid2particle!`, `particle2centroid!`
- Advection: `advection!`, `advection_LinP!`, `advection_MQS!`, `semilagrangian_advection!`, `semilagrangian_advection_LinP!`, `semilagrangian_advection_MQS!`
- Marker-chain utilities: `init_markerchain`, `fill_chain_from_chain!`, `fill_chain_from_vertices!`, `advect_markerchain!`, `semilagrangian_advection_markerchain!`, `interpolate_velocity_to_markerchain!`, `compute_topography_vertex!`, `resample!`
- Phase ratios and diffusion: `PhaseRatios`, `update_phase_ratios!`, `SubgridDiffusionCellArrays`, `subgrid_diffusion!`, `subgrid_diffusion_centroid!`
- Integrators: `Euler`, `RungeKutta2`, `RungeKutta4`
- Checkpointing: `checkpointing_particles`

## Finding Method Docs

For exact signatures and the most up-to-date method documentation, use Julia help mode:

```julia
?init_particles
?advection!
?checkpointing_particles
```
