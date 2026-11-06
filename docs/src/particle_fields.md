# Particle fields

Particle fields use the same cell-local storage as particle coordinates. Create
them with `init_cell_arrays` so their slots and backend match `particles`:

```julia
pT, pρ, = init_cell_arrays(particles, Val(2))
```

`init_cell_arrays(particles, Val(N))` returns a tuple of `N` scalar fields. The
transfer routines accept such tuples directly, which is convenient for vector
quantities:

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

For per-cell data that is not one value per particle slot, such as one value per
material phase in each cell, allocate the `CellArray` with `cell_array`:

```julia
n_phases = 3
phase_data = cell_array(backend, 0.0, (n_phases,), size(particles.index))
```

See [CellArrays](CellArrays.md) for indexing and halo exchange, and
[`to_cpu`](@ref) for copying fields to host memory.
