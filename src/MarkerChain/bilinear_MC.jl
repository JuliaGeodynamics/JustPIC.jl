using Statistics

"""
    compute_topography_vertex!(chain::MarkerChain)

Interpolate the marker-chain geometry back to the vertex-based topography array
`chain.h_vertices`.

This is typically called after marker advection or resampling.
"""
function compute_topography_vertex!(chain::MarkerChain)
    (; coords, index, cell_vertices, h_vertices) = chain
    chain_x, chain_y = coords

    launch!(
        ka_backend(h_vertices), _compute_h_vertex!, length(cell_vertices),
        h_vertices, chain_x, chain_y, cell_vertices, index
    )

    return nothing
end

@kernel function _compute_h_vertex!(
        h_vertices, chain_x, chain_y, cell_vertices, index
    )
    ivertex = @index(Global)
    _compute_h_vertex_kernel!(
        h_vertices, chain_x, chain_y, cell_vertices, index, ivertex
    )
end

function _compute_h_vertex_kernel!(
        h_vertices, chain_x, chain_y, cell_vertices, index, ivertex
    )
    xcorner = cell_vertices[ivertex]
    h = zero(xcorner)
    n = 0

    for j in (ivertex - 1):ivertex
        if 0 < j < length(cell_vertices)
            h_cell, isvalid = _fit_cell_height(chain_x, chain_y, index, j, xcorner)
            h += ifelse(isvalid, h_cell, zero(h_cell))
            n += isvalid
        end
    end
    iszero(n) || (h_vertices[ivertex] = h / n)

    return nothing
end

@inline function _fit_cell_height(chain_x, chain_y, index, icell, xq)
    n = 0
    sd = zero(xq)
    sy = zero(xq)
    sdd = zero(xq)
    sdy = zero(xq)
    for ip in cellaxes(index)
        CAI.@index(index[ip, icell]) || continue
        x = CAI.@index chain_x[ip, icell]
        y = CAI.@index chain_y[ip, icell]
        d = x - xq
        n += 1
        sd += d
        sy += y
        sdd += d * d
        sdy += d * y
    end

    iszero(n) && return zero(xq), false
    dmean = sd / n
    ymean = sy / n
    denominator = sdd - sd * dmean
    slope = ifelse(iszero(denominator), zero(xq), (sdy - sd * ymean) / denominator)
    return muladd(-slope, dmean, ymean), true
end

"""
    mean_height(chain::MarkerChain)

Return the domain mean of the piecewise-linear vertex topography.
"""
function mean_height(h::AbstractVector)
    n = length(h)
    endpoints = sum(@view h[1:1]) + sum(@view h[n:n])
    return (n * mean(h) - endpoints / 2) / (n - 1)
end

mean_height(chain::MarkerChain) = mean_height(chain.h_vertices)

######################################

"""
    reconstruct_chain_from_vertices!(chain::MarkerChain)

Rebuild the markers of each column by evenly distributing them along the straight segment
joining the two bounding `h_vertices`.

The number of markers per column is preserved (empty slots are skipped, so the column need
not be contiguously packed). This is the inverse of [`compute_topography_vertex!`](@ref) and
is used after the vertex topography has been modified (e.g. by the mass-conservation step of
[`advect_markerchain!`](@ref) or the slope limiting of
[`semilagrangian_advection_markerchain!`](@ref)).
"""
function reconstruct_chain_from_vertices!(chain::MarkerChain)
    (; coords, index, cell_vertices, h_vertices) = chain
    chain_x, chain_y = coords

    launch!(
        ka_backend(index), _reconstruct_h_from_vertex!, length(index),
        h_vertices, chain_x, chain_y, cell_vertices, index
    )

    return nothing
end

@kernel function _reconstruct_h_from_vertex!(
        h_vertices, chain_x, chain_y, cell_vertices, index
    )
    ivertex = @index(Global)
    _reconstruct_h_from_vertex_kernel!(
        h_vertices, chain_x, chain_y, cell_vertices, index, ivertex
    )
end

function _reconstruct_h_from_vertex_kernel!(
        h_vertices, chain_x, chain_y, cell_vertices, index, ivertex
    )
    xcorner_left = cell_vertices[ivertex]
    xcorner_right = cell_vertices[ivertex + 1]
    lx = xcorner_right - xcorner_left
    ycorner_left = h_vertices[ivertex]
    ycorner_right = h_vertices[ivertex + 1]
    ly = ycorner_right - ycorner_left

    # count active particles; slots need not be contiguous (move_particles! can leave
    # interior holes), so skip inactive slots instead of stopping at the first one
    np = 0
    for ip in cellaxes(index)
        CAI.@index(index[ip, ivertex]) && (np += 1)
    end

    Δx = lx / (np + 1)
    Δy = ly / (np + 1)
    xp_new = xcorner_left
    yp_new = ycorner_left

    # fill cell arrays with new particle coordinates
    for ip in cellaxes(index)
        CAI.@index(index[ip, ivertex]) || continue

        xp_new += Δx
        yp_new += Δy
        CAI.@index chain_x[ip, ivertex] = xp_new
        CAI.@index chain_y[ip, ivertex] = yp_new
    end

    return nothing
end

"""
    smooth_slopes!(chain::MarkerChain, max_angle::Real)

Limit the local slope of the vertex topography to `max_angle` (in radians).

Interior vertices whose left or right slope steeper than `tan(max_angle)` are replaced by a
3-point average of themselves and their neighbours (LaMEM-style limiting). This suppresses
the spurious spikes that semi-Lagrangian backtracking can introduce on a steep interface;
it is applied automatically inside [`semilagrangian_advection_markerchain!`](@ref). Chains
with fewer than three vertices are left untouched.
"""
function smooth_slopes!(chain::MarkerChain, max_angle::Real)
    (; h_vertices, cell_vertices) = chain
    n = length(h_vertices)

    n < 3 && return nothing  # Need at least 3 vertices for smoothing

    tan_max_angle = convert(eltype(h_vertices), tan(max_angle))

    h_smoothed = similar(h_vertices)
    copyto!(h_smoothed, h_vertices)  # Initialize with original values

    launch!(
        ka_backend(h_vertices), smooth_slopes_kernel!, n - 2,
        h_smoothed, h_vertices, cell_vertices, tan_max_angle
    )

    # Copy results back
    copyto!(h_vertices, h_smoothed)
    return nothing
end

@kernel function smooth_slopes_kernel!(
        h_smoothed, h_vertices, cell_vertices, tan_max_angle
    )
    i0 = @index(Global)
    i = i0 + 1

    # Each thread handles one vertex independently
    dx_left = cell_vertices[i] - cell_vertices[i - 1]
    dx_right = cell_vertices[i + 1] - cell_vertices[i]
    dh_left = h_vertices[i] - h_vertices[i - 1]
    dh_right = h_vertices[i + 1] - h_vertices[i]

    slope_left = abs(dh_left / dx_left)
    slope_right = abs(dh_right / dx_right)

    # If either adjacent slope is too steep, apply smoothing
    if slope_left > tan_max_angle || slope_right > tan_max_angle
        # Simple 3-point averaging (integer literals keep the array's precision)
        h_smoothed[i] = (h_vertices[i - 1] + 2 * h_vertices[i] + h_vertices[i + 1]) / 4
    else
        # Keep original value
        h_smoothed[i] = h_vertices[i]
    end
end
