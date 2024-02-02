function resample!(p::MarkerChain)
    @parallel_indices (i) function resample!(
        coords, cell_vertices, index, min_xcell, max_xcell, dx_cells
    )

        # cell particles coordinates
        x_cell, y_cell = coords[1][i], coords[2][i]
        # number of particles in the cell
        np = count(index_cell)
        # dx of the new chain
        dx_chain = dx_cells / (np + 1)
        # resample the cell if the number of particles is  
        # less than min_xcell or it is too distorted
        do_resampling = (np < min_xcell) * isdistorded(x_cell, dx_chain)

        if do_resampling
            # lower-left corner of the cell
            x0 = cell_vertices[i]
            # fill index array
            for ip in 1:nxcell
                # x query point
                @cell px[ip, i] = xq = x0 + dx_chain * ip
                # interpolated y coordinated
                yq = if 1 < i < length(x_cell)
                    # inner cells; this is true (ncells-2) consecutive times
                    interp1D_inner(xq, coords, i)
                else
                    # first and last cells
                    interp1D_extremas(xq, x_cell, y_cell)
                end
                @cell py[ip, i] = yq
                @cell index[ip, i] = true
            end
            # fill empty memory locations
            for ip in (nxcell + 1):max_xcell
                @cell px[ip, i] = NaN
                @cell py[ip, i] = NaN
                @cell index[ip, i] = false
            end
        end
        return nothing
    end

    (; coords, cell_vertices, min_xcell, max_xcell) = p
    nx = length(x) - 1
    dx_cells = cell_length(chain)

    # call kernel
    @parallel (1:nx) resample!(coords, cell_vertices, index, min_xcell, max_xcell, dx_cells)
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
        # check wether next memory location holds a particle;
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
