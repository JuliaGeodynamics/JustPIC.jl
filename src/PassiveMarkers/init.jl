"""
    init_passive_markers(backend, coords::NTuple{N,AbstractArray})

Construct a [`PassiveMarkers`](@ref) container on `backend` from marker
coordinate arrays.

`coords` is an `N`-tuple of vectors, one per spatial dimension, holding the
initial marker positions: marker `k` sits at `(coords[1][k], …, coords[N][k])`.

# Arguments
- `backend`: KernelAbstractions backend type such as `CPU`.
- `coords`: tuple of coordinate vectors, one per dimension.
"""
function init_passive_markers(backend, coords::NTuple{N, AbstractArray}) where {N}
    return PassiveMarkers(backend, coords)
end
