function init_passive_markers(backend, coords::NTuple{N, AbstractArray}) where N
    
    @parallel_indices (i) function fill_coords_index!(pxᵢ::NTuple{N, AbstractArray}, coords::NTuple{N, AbstractArray}) where N
        # fill index array
        ntuple(Val(N)) do dim
            for i in cellaxes(pxᵢ[dim])
                @cell pxᵢ[dim][i, 1] = coords[dim][i]
            end
        end
        return nothing
    end

    np = length(coords[1])

    pxᵢ = ntuple(_ -> @fill(0.0, (1,), celldims = (np,)), Val(N))
   
    @parallel (1:np) fill_coords_index!(pxᵢ, coords)

    return PassiveMarkers(backend, pxᵢ)
end