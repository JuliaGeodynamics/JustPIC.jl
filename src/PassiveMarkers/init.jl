function init_passive_markers(backend, coords::NTuple{N, AbstractArray}) where {N}
    return PassiveMarkers(backend, coords)
end
