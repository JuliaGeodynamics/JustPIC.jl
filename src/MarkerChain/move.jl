"""
    move_particles!(chain::MarkerChain)

Reassign markers to the correct columns of `chain` after their coordinates have
been updated.

Markers that crossed column boundaries are moved into their destination column's
slots, keeping the coordinate arrays consistent with the per-column occupancy
mask. A marker may cross any number of columns in one call. Markers whose updated
coordinates are not finite, or which left the horizontal extent of
`chain.cell_vertices`, are deleted.
"""
function move_particles!(chain::MarkerChain)
    (; coords, index, cell_vertices) = chain
    nxi = size(index, 1)
    grid = cell_vertices

    cell_jumps = similar(chain.h_vertices, Int, nxi)
    launch!(
        ka_backend(index), maximum_cell_jump!, nxi,
        cell_jumps, coords, grid, index
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
            coords, grid, index, offset, nxi, ncolors
        )
    end

    return nothing
end

@inline in_column(x, grid, i::Integer) = grid[i] ≤ x < grid[i + 1]

# The cell lookup clamps, so it cannot signal that a marker left the domain; this test has
# to run first.
@inline outside_grid(x, grid) = x < first(grid) || x ≥ last(grid)

@kernel function maximum_cell_jump!(cell_jumps, coords, grid, index)
    i = @index(Global)
    max_jump = 0

    for ip in cellaxes(index)
        doskip(index, ip, i) && continue
        pᵢ = cache_particle(coords, ip, i)
        (!isfinite(pᵢ[1]) || !isfinite(pᵢ[2])) && continue
        in_column(pᵢ[1], grid, i) && continue
        outside_grid(pᵢ[1], grid) && continue

        new_cell = parent_cell_index(pᵢ[1], grid, i)
        max_jump = max(max_jump, abs(new_cell - i))
    end

    cell_jumps[i] = max_jump
end

@kernel function move_particles_launcher!(coords, grid, index, offset, nxi, ncolors)
    i0 = @index(Global)
    i = ncolors * (i0 - 1) + offset
    i ≤ nxi && _move_particles!(coords, grid, index, i)
end

function _move_particles!(coords, grid, index, idx)
    # iterate over particles in child cell
    for ip in cellaxes(index)
        doskip(index, ip, idx) && continue
        pᵢ = cache_particle(coords, ip, idx)

        if !isfinite(pᵢ[1]) || !isfinite(pᵢ[2]) || outside_grid(pᵢ[1], grid)
            ## SOMEHOW THE PARTICLE DID ESCAPE THE DOMAIN
            ## => REMOVE IT
            @inbounds CAI.@index index[ip, idx] = false
            empty_particle!(coords, ip, idx)

        else
            # check whether the particle is
            # within the same cell and skip it
            in_column(pᵢ[1], grid, idx) && continue

            # new cell index, seeded with the column the marker is leaving
            new_cell = parent_cell_index(pᵢ[1], grid, idx)

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
        end
    end
    return nothing
end
