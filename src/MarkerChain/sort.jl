function sort_chain!(p::MarkerChain{B,N}) where {B,N}
    (; coords) = p
    # sort permutations of each cell
    ni = size(first(coords))
    @parallel (@idx ni) _sort!(coords)
end

# 1D MarkerChain 
@parallel_indices (I...) function _sort!(coords::NTuple{2,T}) where {T}

    # extract and save cell particles coordinates
    particle_xᵢ = ntuple(Val(2)) do i
        coords[i][I...]
    end

    # sort permutations of each cell
    permutations = sortperm(first(particle_xᵢ))

    # if cell is already sorted, do nothing
    if !issorted(permutations)
        # otherwise, sort the cell
        for ip in eachindex(permutations)
            permutationᵢ = permutations[ip]
            @assert permutationᵢ ≤ length(permutations)
            ntuple(Val(2)) do i
                @cell coords[i][ip, I...] = particle_xᵢ[i][permutationᵢ]
            end
        end
    end

    return nothing
end
