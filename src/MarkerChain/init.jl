function init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, initial_elevation)
    @parallel_indices (i) function fill_coords_index!(
        px, py, index, x, initial_elevation, dx_chain, nxcell, max_xcell
    )
        # lower-left corner of the cell
        x0 = x[i]
        # fill index array
        for ip in 1:nxcell
            @cell px[ip, i] = x0 + dx_chain * ip
            @cell py[ip, i] = initial_elevation
            @cell index[ip, i] = true
        end
        return nothing
    end

    nx = length(x) - 1
    dx = xv[2] - xv[1]
    dx_chain = dx / (nxcell + 1)
    px, py = ntuple(_ -> @fill(NaN, (nx,), celldims = (max_xcell,)), Val(2))
    index = @fill(false, (nx,), celldims = (max_xcell,), eltype = Bool)

    @parallel (1:nx) fill_coords_index!(
        px, py, index, xv, initial_elevation, dx_chain, nxcell, max_xcell
    )

    return MarkerChain(backend, (px, py), index, xv, min_xcell, max_xcell)
end
