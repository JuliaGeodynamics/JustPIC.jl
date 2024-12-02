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

## fill chain with given topo 

"""
    fill_chain!(chain::MarkerChain, topo_x, topo_y)

Fill the given `chain` of markers with topographical data.

# Arguments
- `chain::MarkerChain`: The chain of markers to be filled.
- `topo_x`: The x-coordinates of the topography.
- `topo_y`: The y-coordinates of the topography.

# Description
This function populates the `chain` with markers based on the provided topographical data (`topo_x` and `topo_y`). The function modifies the `chain` in place.
"""
function fill_chain!(chain::MarkerChain, topo_x, topo_y)

    (; coords, index, cell_vertices) = chain
    @parallel (1:length(index)) _fill_chain!(coords, index, cell_vertices, topo_x, topo_y)

    return nothing
end

@parallel_indices (icell) function _fill_chain!(coords, index, cell_vertices, topo_x, topo_y)
    _fill_chain_kernel!(coords, index, cell_vertices, topo_x, topo_y, icell)
    return nothing
end

function _fill_chain_kernel!(coords, index, cell_vertices, topo_x, topo_y, icell)

    itopo, ilast = first_last_particle_incell(topo_x, cell_vertices, icell)

    for ip in cellaxes(index)

        if itopo ≤ ilast
            @index index[ip, icell] = true
            @index coords[1][ip, icell] = topo_x[itopo]
            @index coords[2][ip, icell] = topo_y[itopo]
            itopo += 1
        else
            @index index[ip, icell] = false
            @index coords[1][ip, icell] = NaN
            @index coords[2][ip, icell] = NaN
        end

        !(@index index[ip+1, icell]) && break

    end

    return nothing
end

function first_last_particle_incell(topo_x, cell_vertices, icell)
    
    xlims = cell_vertices[icell], cell_vertices[icell + 1]

    ifirst = 1
    ilast  = length(topo_x)
    x1     = topo_x[1]
    previous_incell = xlims[1] < x1 < xlims[2]

    first_found = false
    last_found  = false

    x = topo_x[2]
    for i in 2:ilast-1
        incell = xlims[1] < x < xlims[2]

        if !previous_incell && incell
            ifirst = i
            first_found = true
        end

        xnext = topo_x[i+1]
        next_incell = xlims[1] < xnext < xlims[2]
        if incell && !next_incell
            ilast = i
            last_found = true
        end

        first_found * last_found && break

        x, previous_incell = xnext, incell
    end

    return ifirst, ilast

end