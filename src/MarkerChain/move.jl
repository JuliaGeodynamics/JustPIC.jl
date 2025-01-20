function move_particles!(chain::MarkerChain)
    (; coords, index, cell_vertices) = chain
    dxi = compute_dx(cell_vertices)
    nxi = size(index, 1)
    grid = cell_vertices

    @parallel (@idx nxi) move_particles_launcher!(coords, grid, dxi, index)

    return nothing
end

@parallel_indices (i) function move_particles_launcher!(coords, grid, dxi, index)
    _move_particles!(coords, grid, dxi, index, i)
    return nothing
end

chop(I::NTuple{2, T}) where {T} = I[1]
chop(I::NTuple{3, T}) where {T} = I[1], I[2]

function _move_particles!(coords, grid, dxi, index, idx)
    # coordinate of the lower-most-left coordinate of the parent cell
    corner_xi = corner_coordinate(grid, idx)

    # iterate over particles in child cell
    for ip in cellaxes(index)
        doskip(index, ip, idx) && continue
        pᵢ = cache_particle(coords, ip, idx)

        # check whether the particle is
        # within the same cell and skip it
        isincell(chop(pᵢ), corner_xi, dxi) && continue

        # new cell index
        new_cell = cell_index(chop(pᵢ), grid, dxi)

        if !(any(<(1), new_cell) || any(new_cell .> length(grid)))
            ## THE PARTICLE DID NOT ESCAPE THE DOMAIN
            # remove particle from child cell
            @inbounds @index index[ip, idx] = false
            @inbounds @index coords[1][ip, idx] = NaN
            @inbounds @index coords[2][ip, idx] = NaN
            # check whether there's empty space in parent cell
            free_idx = find_free_memory(index, new_cell...)
            iszero(free_idx) && continue
            # move particle and its fields to the first free memory location
            @inbounds @index index[free_idx, new_cell] = true
            fill_particle!(coords, pᵢ, free_idx, new_cell)

        else
            ## SOMEHOW THE PARTICLE DID ESCAPE THE DOMAIN
            ## => REMOVE IT
            @inbounds @index index[ip, idx] = false
            empty_particle!(coords, ip, idx)
        end
    end
    return nothing
end
