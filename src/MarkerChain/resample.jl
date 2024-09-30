function resample!(chain::MarkerChain)

    # resampling launch kernel
    @parallel_indices (i) function resample!(
        coords, cell_vertices, index, min_xcell, max_xcell, dx_cells
    )
        resample_cell!(coords, cell_vertices, index, min_xcell, max_xcell, dx_cells, i)
        return nothing
    end

    (; coords, index, cell_vertices, min_xcell, max_xcell) = chain
    nx = length(index)
    dx_cells = cell_length(chain)

    # call kernel
    @parallel (1:nx) resample!(coords, cell_vertices, index, min_xcell, max_xcell, dx_cells)
    return nothing
end

function resample_cell!(
    coords::NTuple{2,T}, cell_vertices, index, min_xcell, max_xcell, dx_cells, I
) where {T}

    # cell particles coordinates
    index_I = @cell index[I]
    px, py = coords[1], coords[2]
    x_cell = @cell px[I]
    y_cell = @cell py[I]

    # sort particles in the cell
    perms = sortperm(x_cell)
    x_cell = x_cell[perms]
    y_cell = y_cell[perms]
    index_I = index_I[perms]

    @cell index[I] = index_I
    @cell px[I] = x_cell
    @cell py[I] = y_cell

    # lower-left corner of the cell
    cell_vertex = cell_vertices[I]
    # number of particles in the cell
    np = count(index_I)
    # dx of the new chain
    dx_chain = dx_cells / (np + 1)
    # resample the cell if the number of particles is
    # less than min_xcell or it is too distorted
    do_resampling = (np < min_xcell) || isdistorded(x_cell, dx_chain)

    np_new = max(min_xcell, np)
    dx_chain = dx_cells / (np_new + 1)
    if do_resampling
        # @show I
        # fill index array
        for ip in 1:np_new
            # x query point
            @index px[ip, I] = xq = cell_vertex + dx_chain * ip
            # interpolated y coordinated
            yq = if 1 < I < length(index)
                # inner cells; this is true (ncells-2) consecutive times
                yq = interp1D_inner(xq, x_cell, y_cell, coords, I)
                if isnan(yq)
                    @show xq
                    @show x_cell
                    @show y_cell
                    @show I
                    error("BOOM 1")
                end
                yq
            else
                # first and last cells
                yq = interp1D_extremas(xq, x_cell, y_cell)
                if isnan(yq)
                    error("BOOM 1")
                end

                yq
            end
            if isnan(yq)
                error("BOOM")
            end
            @index py[ip, I] = yq
            @index index[ip, I] = true
        end
        # fill empty memory locations
        for ip in (np_new + 1):max_xcell
            @index px[ip, I] = NaN
            @index py[ip, I] = NaN
            @index index[ip, I] = false
        end
    end
    return nothing
end

function isdistorded(x_cell, dx_ideal)
    for ip in eachindex(x_cell)[1:(end - 1)]
        # current particle
        current_x = x_cell[ip]
        # if there is no particle in this memory location,
        # we do nothing
        isnan(current_x) && continue
        # next particle
        next_x = x_cell[ip + 1]
        # check whether next memory location holds a particle;
        # if thats the case, find the next particle
        if isnan(next_x)
            next_index = findnext(!isnan, x_cell, ip + 1)
            isnothing(next_index) && break
            next_x = x_cell[next_index]
        end
        # check if the distance between particles is greater than 2*dx_ideal
        # if so, return true so that the cell is resampled
        dx = next_x - current_x
        if dx > 2 * dx_ideal
            return true
        end
    end
    return false
end
