# Particle fields

Particle fields use the same cell-local storage as particle coordinates. Create
them with `init_cell_arrays` so their slots and backend match `particles`:

```julia
pT, pρ, = init_cell_arrays(particles, Val(2))
```

The returned fields can be scalar or tuple-valued. Tuple allocation is useful
for vector quantities:

```julia
pV = init_cell_arrays(particles, Val(2))
particle2grid!((Vx, Vy), pV, particles)
```

Pass companion fields through movement and injection routines so values stay
attached to their particles:

```julia
move_particles!(particles, (pT, pρ))
inject_particles!(particles, (pT, pρ))
```

Only active particle slots contain valid values. Unused slots are represented
by the occupancy/index mask and must not be included in reductions.

For a field with explicit cell-array dimensions, use `cell_array`:

```julia
phase = cell_array(backend, zero(Float32), (nphases,), size(particles.index))
```

See [CellArrays](CellArrays.md) for halo exchange, indexing, and conversion
details.
