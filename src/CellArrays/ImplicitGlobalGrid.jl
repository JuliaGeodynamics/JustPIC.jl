# """
#     update_cell_halo!(x::CellArray...)

# Synchronize the overlapping MPI halo of one or more `CellArray`s in place.

# This is the `CellArray` companion to `ImplicitGlobalGrid.update_halo!` and is
# typically used after particle coordinates or per-particle fields have changed on
# each rank.

# # Arguments
# - `x`: one or more `CellArray`s with the same logical grid layout.

# # Notes
# - Every provided `CellArray` is updated; this is convenient for
#   `particles.coords`, `particles.index`, and particle field arrays returned by
#   `init_cell_arrays`.
# - For MPI particle advection, halo exchange is usually required before
#   `move_particles!` so that particles that crossed a rank boundary are visible
#   to the neighboring rank.
# - With periodic boundary conditions, `update_cell_halo!` exchanges the overlap
#   across the periodic domain boundaries as configured in `init_global_grid`.
# - If particles are reinjected with `inject_particles!`, refresh the halos again
#   before reconstructing grid fields with `particle2grid!`.

# # Example
# ```julia
# advection!(particles, RungeKutta2(), V, dt)
# update_cell_halo!(particles.coords...)
# update_cell_halo!(particle_args...)
# update_cell_halo!(particles.index)
# move_particles!(particles, particle_args)
# inject_particles!(particles, particle_args)
# particle2grid!(T, pT, particles)
# ```
# """
# function update_cell_halo!(x::Vararg{CellArray, N}) where {N}
#     for xᵢ in x
#         update_halo!(xᵢ)
#     end
#     return
# end

function update_cell_halo!(
        x::Vararg{CellArray{S, N, D, A}, NA}
    ) where {NA, S, N, D, A <: AbstractArray}
    ni = size(x[1])
    tmp = @fill(0.0e0, ni..., eltype = eltype(x[1].data))
    for xᵢ in x
        for ip in cellaxes(xᵢ)
            @parallel (@idx ni) move_CellArray_to_Array!(tmp, xᵢ, ip)
            update_halo!(tmp)
            @parallel (@idx ni) move_Array_to_CellArray!(xᵢ, tmp, ip)
        end
        return nothing
    end
    return
end

@parallel_indices (I...) function move_Array_to_CellArray!(A::CellArray, B::AbstractArray, ip)
    @inbounds @index A[ip, I...] = B[I...]
    return nothing
end

@parallel_indices (I...) function move_CellArray_to_Array!(B::AbstractArray, A::CellArray, ip)
    @inbounds B[I...] = @index A[ip, I...]
    return nothing
end