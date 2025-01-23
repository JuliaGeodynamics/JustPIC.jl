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
