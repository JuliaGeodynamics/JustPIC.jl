function sort_chain!(p::MarkerChain)
    (; coords, index) = p
    # sort permutations of each cell
    ni = size(first(coords))
    @parallel (@idx ni) _sort!(coords, index)
end

# 1D MarkerChain 
@parallel_indices (I...) function _sort!(coords::NTuple{2,T}, index) where {T}

    # extract and save cell particles coordinates
    particle_xᵢ = ntuple(Val(2)) do i
        coords[i][I...]
    end
    indexᵢ = index[I...]

    # sort permutations of each cell
    permutations = sortperm(first(particle_xᵢ))

    # if cell is already sorted, do nothing
    if !issorted(permutations)
        # otherwise, sort the cell
        for ip in eachindex(permutations)
            permutationᵢ = permutations[ip]
            @assert permutationᵢ ≤ length(permutations)

            @cell coords[1][ip, I...] = particle_xᵢ[1][permutationᵢ]
            @cell coords[2][ip, I...] = particle_xᵢ[2][permutationᵢ]
            @cell index[ip, I...] = indexᵢ[permutationᵢ]
        end
    end

    return nothing
end
