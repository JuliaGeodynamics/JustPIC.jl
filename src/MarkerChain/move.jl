"""
    move_particles!(chain::MarkerChain)

Reassign markers to the correct columns of `chain` after their coordinates have
been updated.

Markers that crossed column boundaries are moved into their destination column's
slots, keeping the coordinate arrays consistent with the per-column occupancy
mask. A marker may cross any number of columns in one call. The horizontal grid
and spacing are taken from `chain.cell_vertices`.
"""
function move_particles!(chain::MarkerChain)
    (; coords, index, cell_vertices) = chain
    dxi = cell_length(chain)
    nxi = size(index, 1)
    grid = cell_vertices

    cell_jumps = similar(chain.h_vertices, Int, nxi)
    launch!(
        ka_backend(index), maximum_cell_jump!, nxi,
        cell_jumps, coords, grid, dxi, index, nxi
    )
    max_jump = maximum(cell_jumps)

    # Sources of the same color are farther apart than the diameter of their
    # possible destination intervals. They therefore cannot write to the same
    # cell, even when markers cross multiple columns in one timestep.
    ncolors = 2 * max_jump + 1
    n_color_cells = cld(nxi, ncolors)
    for offset in 1:min(ncolors, nxi)
        launch!(
            ka_backend(index), move_particles_launcher!, n_color_cells,
            coords, grid, dxi, index, offset, nxi, ncolors
        )
    end

    return nothing
end

@kernel function maximum_cell_jump!(cell_jumps, coords, grid, dxi, index, nxi)
    i = @index(Global)
    corner_xi = corner_coordinate(grid, i)
    max_jump = 0

    for ip in cellaxes(index)
        doskip(index, ip, i) && continue
        pᵢ = cache_particle(coords, ip, i)
        (!isfinite(pᵢ[1]) || !isfinite(pᵢ[2])) && continue
        isincell(pᵢ[1], corner_xi, dxi) && continue

        new_cell = cell_index(pᵢ[1], grid, dxi)
        if 1 ≤ new_cell ≤ nxi
            max_jump = max(max_jump, abs(new_cell - i))
        end
    end

    cell_jumps[i] = max_jump
end

@kernel function move_particles_launcher!(coords, grid, dxi, index, offset, nxi, ncolors)
    i0 = @index(Global)
    i = ncolors * (i0 - 1) + offset
    i ≤ nxi && _move_particles!(coords, grid, dxi, index, i)
end

function _move_particles!(coords, grid, dxi, index, idx)
    # coordinate of the lower-most-left coordinate of the parent cell
    corner_xi = corner_coordinate(grid, idx)

    # iterate over particles in child cell
    for ip in cellaxes(index)
        doskip(index, ip, idx) && continue
        pᵢ = cache_particle(coords, ip, idx)

        if !isfinite(pᵢ[1]) || !isfinite(pᵢ[2])
            ## SOMEHOW THE PARTICLE DID ESCAPE THE DOMAIN
            ## => REMOVE IT
            @inbounds CAI.@index index[ip, idx] = false
            empty_particle!(coords, ip, idx)

        else
            # check whether the particle is
            # within the same cell and skip it
            isincell(pᵢ[1], corner_xi, dxi) && continue

            # new cell index
            new_cell = cell_index(pᵢ[1], grid, dxi)

            if 1 ≤ new_cell < length(grid)
                ## THE PARTICLE DID NOT ESCAPE THE DOMAIN
                # remove particle from child cell
                nan = convert(eltype(eltype(coords[1])), NaN)
                @inbounds CAI.@index index[ip, idx] = false
                @inbounds CAI.@index coords[1][ip, idx] = nan
                @inbounds CAI.@index coords[2][ip, idx] = nan
                # check whether there's empty space in parent cell
                free_idx = find_free_memory(index, new_cell...)
                iszero(free_idx) && continue
                # move particle and its fields to the first free memory location
                @inbounds CAI.@index index[free_idx, new_cell] = true
                fill_particle!(coords, pᵢ, free_idx, new_cell)

            else
                ## SOMEHOW THE PARTICLE DID ESCAPE THE DOMAIN
                ## => REMOVE IT
                @inbounds CAI.@index index[ip, idx] = false
                empty_particle!(coords, ip, idx)
            end
        end
    end
    return nothing
end
