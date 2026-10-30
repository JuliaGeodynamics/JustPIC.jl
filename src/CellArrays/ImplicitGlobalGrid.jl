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
- The whole overlap is exchanged, so cells held by two neighboring ranks end
  up with identical particles, taken from the rank that owns each cell.
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
function update_cell_halo!(
        x::Vararg{CellArray, NA}
    ) where {NA}
    ImplicitGlobalGrid.grid_is_initialized() || throw(
        ArgumentError("`update_cell_halo!` requires an ImplicitGlobalGrid; call `init_global_grid` first")
    )
    ni = size(x[1])
    backend = ka_backend(x[1])
    halowidths = cell_halowidths(ni)
    for xᵢ in x
        size(xᵢ) == ni || throw(DimensionMismatch("CellArrays must have the same grid size"))
        ka_backend(xᵢ) == backend || throw(ArgumentError("CellArrays must use the same backend"))
        tmp = KernelAbstractions.zeros(backend, eltype(xᵢ.data), ni...)
        for ip in cellaxes(xᵢ)
            launch!(backend, move_CellArray_to_Array_kernel!, ni, tmp, xᵢ, ip)
            update_halo!((A = tmp, halowidths))
            launch!(backend, move_Array_to_CellArray_kernel!, ni, xᵢ, tmp, ip)
        end
    end
    return nothing
end

# Exchange half of the IGG overlap `overlaps + size - nxyz` from each side, so every cell
# held by two ranks has one owner. Particle CellArrays (one ghost per side) get width 2;
# IGG's default width of 1 left the two inner shared cells unsynchronized.
function cell_halowidths(ni::NTuple{N, Integer}) where {N}
    gg = ImplicitGlobalGrid.global_grid()
    return ntuple(d -> max(1, (gg.overlaps[d] + get(ni, d, 1) - gg.nxyz[d]) ÷ 2), Val(3))
end

@kernel function move_Array_to_CellArray_kernel!(A::CellArray, B::AbstractArray, ip)
    I = @index(Global, NTuple)
    CAI.@index A[ip, I...] = B[I...]
end

@kernel function move_CellArray_to_Array_kernel!(B::AbstractArray, A::CellArray, ip)
    I = @index(Global, NTuple)
    B[I...] = CAI.@index A[ip, I...]
end
