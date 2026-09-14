"""
    init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, initial_elevation)

Create a 2D `MarkerChain` sampled along the horizontal grid `xv`.

The vertices in `xv` must be finite and strictly increasing; the spacing may be
non-uniform, in which case each column is populated according to its own width.

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
    _validate_chain_grid(xv)
    nx = length(xv) - 1
    0 < nxcell ≤ max_xcell || throw(ArgumentError("nxcell must satisfy 0 < nxcell ≤ max_xcell"))
    0 < min_xcell ≤ max_xcell || throw(ArgumentError("min_xcell must satisfy 0 < min_xcell ≤ max_xcell"))
    initial_elevation isa AbstractArray && length(initial_elevation) != nx + 1 &&
        throw(DimensionMismatch("initial_elevation must have one value per grid vertex"))
    # a refined grid and a vertex-wise elevation are both indexed inside the kernels below,
    # so they have to live on the device
    xv = device_grid(backend, xv, T)
    initial_elevation = if initial_elevation isa AbstractArray
        TA(backend){T}(initial_elevation)
    else
        convert(T, initial_elevation)
    end
    px, py = ntuple(_ -> cell_array(backend, convert(T, NaN), (max_xcell,), (nx,)), Val(2))
    index = cell_array(backend, false, (max_xcell,), (nx,))

    launch!(
        ka_backend(index), fill_markerchain_coords_index!, nx,
        px, py, index, xv, initial_elevation, nxcell, max_xcell
    )
    coords = px, py
    px0, py0 = ntuple(_ -> cell_array(backend, convert(T, NaN), (max_xcell,), (nx,)), Val(2))
    copyto!(px0.data, px.data)
    copyto!(py0.data, py.data)
    coords0 = px0, py0
    h_vertices = if initial_elevation isa AbstractArray
        copy(initial_elevation)
    else
        TA(backend)(fill(initial_elevation, nx + 1))
    end
    h_vertices0 = copy(h_vertices)

    return MarkerChain(
        backend, coords, coords0, h_vertices, h_vertices0, xv, index, max_xcell, min_xcell
    )
end

@kernel function fill_markerchain_coords_index!(
        px, py, index, x, initial_elevation, nxcell, max_xcell
    )
    i = @index(Global)

    # lower-left corner of the cell
    x0 = x[i]
    dx_chain = cell_width(x, i) / (nxcell + 1)
    # fill index array
    for ip in 1:nxcell
        CAI.@index px[ip, i] = x0 + dx_chain * ip
        CAI.@index py[ip, i] = initial_elevation
        CAI.@index index[ip, i] = true
    end
end

@kernel function fill_markerchain_coords_index!(
        px, py, index, x, initial_elevation::AbstractArray{T, 1}, nxcell, max_xcell
    ) where {T}
    i = @index(Global)

    # lower-left corner of the cell
    x0 = x[i]
    dx_chain = cell_width(x, i) / (nxcell + 1)
    elevation_left = initial_elevation[i]
    elevation_right = initial_elevation[i + 1]
    # fill index array
    for ip in 1:nxcell
        fraction = convert(T, ip) / convert(T, nxcell + 1)
        CAI.@index px[ip, i] = x0 + dx_chain * ip
        CAI.@index py[ip, i] = elevation_left + (elevation_right - elevation_left) * fraction
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
    ilast = 0
    for i in eachindex(topo_x)
        if xlims[1] < topo_x[i] < xlims[2]
            iszero(ilast) && (ifirst = i)
            ilast = i
        elseif !iszero(ilast)
            break
        end
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
