"""
    update_cell_halo!(x::CellArray...)

Synchronize the overlapping MPI halo of one or more `CellArray`s in place.

This is the `CellArray` companion to `ImplicitGlobalGrid.update_halo!` and is
typically used after particle coordinates or per-particle fields have changed on
each rank.

# Arguments
- `x`: one or more `CellArray`s with the same logical grid layout.

# Notes
- Every provided `CellArray` is updated; this is convenient for
  `particles.coords`, `particles.index`, and particle field arrays returned by
  `init_cell_arrays`.
- For MPI particle advection, halo exchange is usually required before
  `move_particles!` so that particles that crossed a rank boundary are visible
  to the neighboring rank.
- With periodic boundary conditions, `update_cell_halo!` exchanges the overlap
  across the periodic domain boundaries as configured in `init_global_grid`.
- If particles are reinjected with `inject_particles!`, refresh the halos again
  before reconstructing grid fields with `particle2grid!`.

# Example
```julia
advection!(particles, RungeKutta2(), V, dt)
update_cell_halo!(particles.coords...)
update_cell_halo!(particle_args...)
update_cell_halo!(particles.index)
move_particles!(particles, particle_args)
inject_particles!(particles, particle_args)
particle2grid!(T, pT, particles)
```
"""
function update_cell_halo!(x::Vararg{CellArray, N}) where {N}
    for xᵢ in x
        update_halo!(xᵢ)
    end
    return
end