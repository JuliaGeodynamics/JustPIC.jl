function init_markerchain(::Type{JustPIC.CPUBackend}, nxcell, min_xcell, max_xcell, xv, initial_elevation)
    nx = length(xv) - 1
    dx = xv[2] - xv[1]
    dx_chain = dx / (nxcell + 1)
    px, py = ntuple(_ -> @fill(NaN, (nx,), celldims = (max_xcell,)), Val(2))
    index = @fill(false, (nx,), celldims = (max_xcell,), eltype = Bool)

    @parallel (1:nx) fill_markerchain_coords_index!(
        px, py, index, xv, initial_elevation, dx_chain, nxcell, max_xcell
    )

    return MarkerChain(JustPIC.CPUBackend, (px, py), index, xv, min_xcell, max_xcell)
end

@parallel_indices (i) function fill_markerchain_coords_index!(
    px, py, index, x, initial_elevation, dx_chain, nxcell, max_xcell
)
    # lower-left corner of the cell
    x0 = x[i]
    # fill index array
    for ip in 1:nxcell
        @index px[ip, i] = x0 + dx_chain * ip
        @index py[ip, i] = initial_elevation
        @index index[ip, i] = true
    end
    return nothing
end

@parallel_indices (i) function fill_markerchain_coords_index!(
    px, py, index, x, initial_elevation::AbstractArray{T, 1}, dx_chain, nxcell, max_xcell
) where {T}
    # lower-left corner of the cell
    x0 = x[i]
    initial_elevation0 = initial_elevation[i]
    # fill index array
    for ip in 1:nxcell
        @index px[ip, i] = x0 + dx_chain * ip
        @index py[ip, i] = initial_elevation0
        @index index[ip, i] = true
    end
    return nothing
end