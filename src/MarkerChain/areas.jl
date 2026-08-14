"""
    compute_rock_fraction!(ratios, chain::MarkerChain, xvi, dxi)

Fill `ratios` with the fraction of each control volume that lies below the marker
chain.

The result is written at cell centers, vertices, and staggered velocity nodes
using the topography currently stored in `chain`.
"""
function compute_rock_fraction!(ratios, chain::MarkerChain, xvi, dxi)
    compute_area_below_chain_centers!(ratios.center, chain, xvi, dxi)
    compute_area_below_chain_vertex!(ratios.vertex, chain, xvi, dxi)
    compute_area_below_chain_vx!(ratios.Vx, chain, xvi, dxi)
    compute_area_below_chain_vy!(ratios.Vy, chain, xvi, dxi)
    return nothing
end

function compute_area_below_chain_centers!(ratio_center, chain, xvi, dxi)
    topo_y = chain.h_vertices
    nx, ny = size(ratio_center)
    launch!(
        ka_backend(ratio_center), _compute_area_below_chain_center!, (nx, ny),
        ratio_center, topo_y, xvi..., dxi
    )
    return nothing
end

@kernel function _compute_area_below_chain_center!(
        ratio::AbstractArray, topo_y, xv, yv, dxi
    )
    I = @index(Global, NTuple)
    i, j = I

    # cell origin
    ox = xv[i]
    oy = yv[j]

    p1 = GridGeometryUtils.Point(ox, topo_y[i])
    p2 = GridGeometryUtils.Point(xv[i + 1], topo_y[i + 1])
    s = Segment(p1, p2)

    r = BBox((ox, oy), dxi...)
    ratio[i, j] = cell_rock_area(s, r)
end

function compute_area_below_chain_vx!(ratio_velocity, chain, xvi, dxi)
    topo_y = chain.h_vertices
    nx, ny = size(ratio_velocity)
    mask_x = (-1, 0) .* dxi[1] ./ 2

    launch!(
        ka_backend(ratio_velocity), _compute_area_below_chain_vx!, (nx, ny),
        ratio_velocity, topo_y, mask_x, xvi..., nx, dxi
    )
    return nothing
end

@kernel function _compute_area_below_chain_vx!(
        ratios::AbstractArray{T}, topo_y, mask_x, xv, yv, nx, dxi
    ) where {T}
    I = @index(Global, NTuple)
    i, j = I

    dx, dy = dxi
    half_dx = dx / 2
    half_dy = dy / 2
    c = 0
    ω = 0 # weight for the average
    tmp = zero(T)
    # we can cache the potential coordinates
    x, y = if 1 < i < nx
        (xv[i - 1], xv[i], xv[i + 1]), (topo_y[i - 1], topo_y[i], topo_y[i + 1])
    elseif i == 1
        (xv[i], xv[i], xv[i + 1]), (topo_y[i], topo_y[i], topo_y[i + 1])
    else
        (xv[i - 1], xv[i], xv[i]), (topo_y[i - 1], topo_y[i], topo_y[i])
    end

    ox = xv[i]
    oy = yv[j]
    for (l, ii) in enumerate((i - 1):i)
        c += 1
        !(0 < ii < nx) && continue

        ω += 1

        ## new origin at the center of the (ii, jj)-th cell
        origin = (ox, oy) .+ (mask_x[c], zero(T))
        ## now we need to interpolate the segment of the chain to the boundaries of the new cell
        # segment of the chain
        p1 = GridGeometryUtils.Point(x[l], y[l])
        p2 = GridGeometryUtils.Point(x[l + 1], y[l + 1])
        # create a line from the two points
        l = Line(p1, p2)
        # evaluate the line at the origin and origin + dx / 2
        y1 = line(l, origin[1])
        y2 = line(l, origin[1] + half_dx)
        # create two points at the boundaries of the new cell
        p1 = GridGeometryUtils.Point(origin[1], y1)
        p2 = GridGeometryUtils.Point(origin[1] + half_dx, y2)
        # and turn them into a segment
        s = Segment(p1, p2)

        ## bounding box of the new cell
        r = BBox(origin, half_dx, half_dy)
        tmp += cell_rock_area(s, r)
    end
    ratios[i, j] = tmp / ω
end

function compute_area_below_chain_vy!(ratio_velocity, chain, xvi, dxi)
    topo_y = chain.h_vertices
    nx, ny = size(ratio_velocity)
    mask_y = (-1, 0) .* dxi[2] ./ 2

    launch!(
        ka_backend(ratio_velocity), _compute_area_below_chain_vy!, (nx, ny),
        ratio_velocity, topo_y, mask_y, xvi..., ny, dxi
    )
    return nothing
end

@kernel function _compute_area_below_chain_vy!(
        ratios::AbstractArray{T}, topo_y, mask_y, xv, yv, ny, dxi
    ) where {T}
    I = @index(Global, NTuple)
    i, j = I

    dx, dy = dxi
    half_dx = dx / 2
    half_dy = dy / 2
    c = 0
    ω = 0 # weight for the average
    tmp = zero(T)
    # we can cache the potential coordinates
    x, y = (xv[i], xv[i + 1]), (topo_y[i], topo_y[i + 1])
    ox = xv[i]
    oy = yv[j]

    for (k, jj) in enumerate((j - 1):j)
        c += 1
        !(0 < jj < ny) && continue

        ω += 1

        ## new origin at the center of the (ii, jj)-th cell
        origin = (ox, oy) .+ (zero(T), mask_y[c])
        ## now we need to interpolate the segment of the chain to the boundaries of the new cell
        # segment of the chain
        p1 = GridGeometryUtils.Point(x[1], y[1])
        p2 = GridGeometryUtils.Point(x[2], y[2])
        # create a line from the two points
        l = Line(p1, p2)
        # evaluate the line at the origin and origin + dx / 2
        y1 = line(l, origin[1])
        y2 = line(l, origin[1] + half_dx)
        # create two points at the boundaries of the new cell
        p1 = GridGeometryUtils.Point(origin[1], y1)
        p2 = GridGeometryUtils.Point(origin[1] + half_dx, y2)
        # and turn them into a segment
        s = Segment(p1, p2)

        ## bounding box of the new cell
        r = BBox(origin, half_dx, half_dy)
        tmp += cell_rock_area(s, r)
    end
    ratios[i, j] = tmp / ω
end

function compute_area_below_chain_vertex!(ratio_vertex, chain, xvi, dxi)
    topo_y = chain.h_vertices
    ni = size(ratio_vertex)
    masks_x = (-1, 0, -1, 0) .* dxi[1] ./ 2
    masks_y = (-1, -1, 0, 0) .* dxi[2] ./ 2

    launch!(
        ka_backend(ratio_vertex), _compute_area_below_chain_vertex!, ni,
        ratio_vertex, topo_y, masks_x, masks_y, xvi..., ni..., dxi
    )
    return nothing
end

@kernel function _compute_area_below_chain_vertex!(
        ratios::AbstractArray{T}, topo_y, masks_x, masks_y, xv, yv, nx, ny, dxi
    ) where {T}
    I = @index(Global, NTuple)
    i, j = I

    dx, dy = dxi
    half_dx = dx / 2
    half_dy = dy / 2
    c = 0 # linear index of the mask
    ω = 0 # weight for the average
    tmp = zero(T)
    # we can cache the potential coordinates
    x, y = if 1 < i < nx
        (xv[i - 1], xv[i], xv[i + 1]), (topo_y[i - 1], topo_y[i], topo_y[i + 1])
    elseif i == 1
        (xv[i], xv[i], xv[i + 1]), (topo_y[i], topo_y[i], topo_y[i + 1])
    else
        (xv[i - 1], xv[i], xv[i]), (topo_y[i - 1], topo_y[i], topo_y[i])
    end

    ox = xv[i]
    oy = yv[j]

    for (k, jj) in enumerate((j - 1):j)
        for (l, ii) in enumerate((i - 1):i)
            c += 1
            !(0 < ii < nx) && continue
            !(0 < jj < ny) && continue

            ω += 1

            ## new origin at the center of the (ii, jj)-th cell
            origin = (ox, oy) .+ (masks_x[c], masks_y[c])
            ## now we need to interpolate the segment of the chain to the boundaries of the new cell
            # segment of the chain
            p1 = GridGeometryUtils.Point(x[l], y[l])
            p2 = GridGeometryUtils.Point(x[l + 1], y[l + 1])
            # create a line from the two points
            l = Line(p1, p2)
            # evaluate the line at the origin and origin + dx / 2
            y1 = line(l, origin[1])
            y2 = line(l, origin[1] + half_dx)
            # create two points at the boundaries of the new cell
            p1 = GridGeometryUtils.Point(origin[1], y1)
            p2 = GridGeometryUtils.Point(origin[1] + half_dx, y2)
            # and turn them into a segment
            s = Segment(p1, p2)

            ## bounding box of the new cell
            r = BBox(origin, half_dx, half_dy)
            tmp += cell_rock_area(s, r)
        end
    end
    ratios[i, j] = tmp / ω
end

#############################

# `y1`, `y2` are the ends of the chain segment measured from the floor of a cell of height
# `h`. The comparisons are exact: a round-off-tolerant one carries an absolute near-zero
# tolerance of `1000 * eps(T)`, which for a `Float32` cell only a few thousand `eps` tall
# swallows a large fraction of the cell.
@inline is_chain_above_cell(y1, y2, h) = y1 ≥ h && y2 ≥ h

@inline is_chain_below_cell(y1::T, y2) where {T} = y1 ≤ zero(T) && y2 ≤ zero(T)

# Endpoint of the chord on the boundary of the cell `[0, l] × [0, h]` at the end whose
# vertical edge sits at `x`: the chain's own height when it crosses that edge, otherwise the
# point where it leaves through the floor or the ceiling. `y_other`/`x_other` are the
# opposite end of the chord.
@inline function chord_endpoint(y::T, x, y_other, x_other, h) where {T}
    y_cut = if y < zero(T)
        zero(T)
    elseif y > h
        h
    else
        return GridGeometryUtils.Point(x, y)
    end
    return GridGeometryUtils.Point(x + (y_cut - y) * (x_other - x) / (y_other - y), y_cut)
end

"""
    cell_rock_area(s::Segment, r::BBox{2}) -> Real

Fraction of the cell `r` lying below the marker chain segment `s`, in `[0, 1]`.

`s` spans the full width of `r`, running left to right, and may leave the cell through its
floor or its ceiling.
"""
@inline function cell_rock_area(s::Segment, r::BBox{2, T}) where {T}
    # Heights above the cell floor. `intersecting_area` needs its endpoints on the boundary
    # of the cell and checks that to a tolerance proportional to the cell size, so the whole
    # geometry is built relative to the cell's south-west corner: absolute coordinates carry
    # a round-off proportional to their own magnitude, which for a cell many widths away
    # from the origin exceeds that tolerance.
    l, h = r.l, r.h
    oy = r.origin[2]
    y1, y2 = s.p1[2] - oy, s.p2[2] - oy

    is_chain_above_cell(y1, y2, h) && return one(T)
    is_chain_below_cell(y1, y2) && return zero(T)

    cell = Rectangle((l / 2, h / 2), l, h; θ = zero(T))
    p1 = chord_endpoint(y1, zero(T), y2, l, h)
    p2 = chord_endpoint(y2, l, y1, zero(T), h)

    return clamp(intersecting_area(p1, p2, cell) / area(cell), zero(T), one(T))
end
