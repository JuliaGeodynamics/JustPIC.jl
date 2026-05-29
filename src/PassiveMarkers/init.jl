"""
    init_passive_markers(backend, coords)

Create a `PassiveMarkers` container from user-supplied coordinates.

Passive markers are lightweight tracers: they only store positions and are
suited for pathline tracking or for sampling fields along trajectories without
maintaining the per-cell particle bookkeeping used by `Particles`.

# Arguments
- `backend`: backend type such as `CPUBackend`.
- `coords`: tuple of coordinate arrays, one array per spatial dimension.

# Returns
- A `PassiveMarkers` object ready to be advanced with `advection!`.
"""
function init_passive_markers(backend, coords::NTuple{N, AbstractArray}) where {N}
    return PassiveMarkers(backend, coords)
end

# function init_passive_markers(backend, coords::NTuple{N, AbstractArray}) where {N}
#     @parallel_indices (i) function fill_coords_index!(
#             pxᵢ::NTuple{N, AbstractArray}, coords::NTuple{N, AbstractArray}
#         ) where {N}
#         # fill index array
#         ntuple(Val(N)) do dim
#             pxᵢ[dim][i] = coords[dim][i]
#         end
#         return nothing
#     end

#     # np = length(coords[1])
#     # pxᵢ = ntuple(_ -> CA(backend, (np,)), Val(N))

#     # @parallel (1:np) fill_coords_index!(pxᵢ, coords)

#     return PassiveMarkers(backend, coords)
#     # return PassiveMarkers(backend, pxᵢ)
# end
