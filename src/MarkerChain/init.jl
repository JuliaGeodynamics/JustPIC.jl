"""
    init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, initial_elevation)

Create a 2D `MarkerChain` sampled along the horizontal grid `xv`.

`nxcell` controls the initial number of markers per cell, while
`initial_elevation` can be either a scalar or a vector specifying the initial
surface height.

# Returns
- A `MarkerChain` whose marker positions, vertex topography, and occupancy masks
  are initialized consistently.
"""
function init_markerchain(
        ::Type{backend}, nxcell, min_xcell, max_xcell, xv, initial_elevation
    ) where {backend}
    nx = length(xv) - 1
    dx = xv[2] - xv[1]
    dx_chain = dx / (nxcell + 1)
    px, py = ntuple(_ -> @fill(NaN, (nx,), celldims = (max_xcell,)), Val(2))
    index = @fill(false, (nx,), celldims = (max_xcell,), eltype = Bool)

    @parallel (1:nx) fill_markerchain_coords_index!(
        px, py, index, xv, initial_elevation, dx_chain, nxcell, max_xcell
    )
    coords = px, py
    coords0 = px, py
    h_vertices = @fill(initial_elevation, nx + 1)
    h_vertices0 = @fill(initial_elevation, nx + 1)

    return MarkerChain(
        backend, coords, coords0, h_vertices, h_vertices0, xv, index, min_xcell, max_xcell
    )
end

@parallel_indices (i) function fill_markerchain_coords_index!(
    px, py, index, x, initial_elevation::Number, dx_chain, nxcell, max_xcell
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
    # fill index array
    for ip in 1:nxcell
        @index px[ip, i] = x0 + dx_chain * ip
        @index py[ip, i] = initial_elevation[i]
        @index index[ip, i] = true
    end
    return nothing
end
