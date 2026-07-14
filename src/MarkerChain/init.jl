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
    T = initial_elevation isa AbstractArray ? promote_type(eltype(xv), eltype(initial_elevation)) : promote_type(eltype(xv), typeof(initial_elevation))
    xv = recast_grid(xv, T)
    nx = length(xv) - 1
    dx = xv[2] - xv[1]
    dx_chain = dx / (nxcell + 1)
    initial_elevation = initial_elevation isa AbstractArray ? convert.(T, initial_elevation) : convert(T, initial_elevation)
    px, py = ntuple(_ -> cell_array(backend, convert(T, NaN), (max_xcell,), (nx,)), Val(2))
    index = cell_array(backend, false, (max_xcell,), (nx,))

    launch!(
        ka_backend(index), fill_markerchain_coords_index!, nx,
        px, py, index, xv, initial_elevation, dx_chain, nxcell, max_xcell
    )
    coords = px, py
    coords0 = px, py
    h_vertices = TA(backend)(initial_elevation isa AbstractArray ? initial_elevation : fill(initial_elevation, nx + 1))
    h_vertices0 = copy(h_vertices)

    return MarkerChain(
        backend, coords, coords0, h_vertices, h_vertices0, xv, index, min_xcell, max_xcell
    )
end

@kernel function fill_markerchain_coords_index!(
        px, py, index, x, initial_elevation, dx_chain, nxcell, max_xcell
    )
    i = @index(Global)

    # lower-left corner of the cell
    x0 = x[i]
    # fill index array
    for ip in 1:nxcell
        CAI.@index px[ip, i] = x0 + dx_chain * ip
        CAI.@index py[ip, i] = initial_elevation
        CAI.@index index[ip, i] = true
    end
end

@kernel function fill_markerchain_coords_index!(
        px, py, index, x, initial_elevation::AbstractArray{T, 1}, dx_chain, nxcell, max_xcell
    ) where {T}
    i = @index(Global)

    # lower-left corner of the cell
    x0 = x[i]
    initial_elevation0 = initial_elevation[i]
    # fill index array
    for ip in 1:nxcell
        CAI.@index px[ip, i] = x0 + dx_chain * ip
        CAI.@index py[ip, i] = initial_elevation0
        CAI.@index index[ip, i] = true
    end
end

## fill chain with given topo

"""
    fill_chain_from_chain!(chain::MarkerChain, topo_x, topo_y)

Replace the marker positions in `chain` with coordinates sampled from an existing
topographic polyline.

After the markers are reassigned, the vertex-based topography stored on the chain
is recomputed and synchronized with `h_vertices0`.

`topo_x` and `topo_y` should describe an open polyline that spans the chain's
horizontal extent.
"""
function fill_chain_from_chain!(chain::MarkerChain, topo_x, topo_y)
    (; coords, index, cell_vertices) = chain
    launch!(ka_backend(index), _fill_chain!, length(index), coords, index, cell_vertices, topo_x, topo_y)

    # update topography at the vertices of the grid
    compute_topography_vertex!(chain)
    copyto!(chain.h_vertices0, chain.h_vertices)

    return nothing
end

@kernel function _fill_chain!(
        coords, index, cell_vertices, topo_x, topo_y
    )
    icell = @index(Global)
    _fill_chain_kernel!(coords, index, cell_vertices, topo_x, topo_y, icell)
end

function _fill_chain_kernel!(coords, index, cell_vertices, topo_x, topo_y, icell)
    itopo, ilast = first_last_particle_incell(topo_x, cell_vertices, icell)
    # NaN in the marker precision (a bare NaN literal is Float64 and breaks Metal)
    nan = convert(eltype(eltype(coords[1])), NaN)

    for ip in cellaxes(index)
        if itopo ≤ ilast
            CAI.@index index[ip, icell] = true
            CAI.@index coords[1][ip, icell] = topo_x[itopo]
            CAI.@index coords[2][ip, icell] = topo_y[itopo]
            itopo += 1
        else
            CAI.@index index[ip, icell] = false
            CAI.@index coords[1][ip, icell] = nan
            CAI.@index coords[2][ip, icell] = nan
        end
    end

    return nothing
end

function first_last_particle_incell(topo_x, cell_vertices, icell)
    xlims = cell_vertices[icell], cell_vertices[icell + 1]

    ifirst = 1
    ilast = length(topo_x)
    x1 = topo_x[1]
    previous_incell = xlims[1] < x1 < xlims[2]

    first_found = false
    last_found = false

    x = topo_x[2]
    for i in 2:(ilast - 1)
        incell = xlims[1] < x < xlims[2]

        if !previous_incell && incell
            ifirst = i
            first_found = true
        end

        xnext = topo_x[i + 1]
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

"""
    fill_chain_from_vertices!(chain::MarkerChain, topo_y)

Reconstruct a marker chain from topography values given at grid vertices.

`topo_y` is copied into both the current and previous vertex topography fields
before the marker coordinates are rebuilt.

This is useful when the interface is naturally represented on the vertex grid and
you want to refresh the marker representation from that discretization.
"""
function fill_chain_from_vertices!(chain::MarkerChain, topo_y)
    copyto!(chain.h_vertices, topo_y)
    copyto!(chain.h_vertices0, topo_y)

    # reconstruct marker chain
    reconstruct_chain_from_vertices!(chain)

    # fill also the marker chain from the previous time step
    copyto!(chain.coords0[1].data, chain.coords[1].data)
    copyto!(chain.coords0[2].data, chain.coords[2].data)

    return nothing
end
