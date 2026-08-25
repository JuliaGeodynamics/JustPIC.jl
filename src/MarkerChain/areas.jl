"""
    compute_rock_fraction!(ratios, chain::MarkerChain, xvi, dxi)

Fill `ratios` with the fraction of each control volume that lies below the marker
chain.

The result is written at cell centers, vertices, and staggered velocity nodes
using the topography currently stored in `chain`.

`xvi` are the cell vertices per direction and `dxi` the matching spacings: either one
`Number` per direction for a uniform grid, or one `AbstractVector` of per-cell widths per
direction for a refined one.
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

@inline function rectangle_from_min_corner(min_corner, di)
    center = min_corner[1] + di[1] / 2, min_corner[2] + di[2] / 2
    return Rectangle(center, di...; θ = zero(center[1]))
end

@kernel function _compute_area_below_chain_center!(
        ratio::AbstractArray, topo_y, xv, yv, dxi
    )
    I = @index(Global, NTuple)
    i, j = I

    # cell origin
    ox = xv[i]
    oy = yv[j]
    dx, dy = @dxi dxi i j

    p1 = GridGeometryUtils.Point(ox, topo_y[i])
    p2 = GridGeometryUtils.Point(xv[i + 1], topo_y[i + 1])
    s = Segment(p1, p2)

    # Rectangle's first argument is its center, not its lower-left corner.
    r = Rectangle((ox + dx / 2, oy + dy / 2), dx, dy; θ = zero(ox))
    ratio[i, j] = cell_rock_area(s, r)
end

function compute_area_below_chain_vx!(ratio_velocity, chain, xvi, dxi)
    topo_y = chain.h_vertices
    nx, ny = size(ratio_velocity)

    launch!(
        ka_backend(ratio_velocity), _compute_area_below_chain_vx!, (nx, ny),
        ratio_velocity, topo_y, xvi..., nx, dxi
    )
    return nothing
end

@kernel function _compute_area_below_chain_vx!(
        ratios::AbstractArray{T}, topo_y, xv, yv, nx, dxi
    ) where {T}
    I = @index(Global, NTuple)
    i, j = I

    dy = @dy dxi j
    ω = zero(T) # total area of the sampled sub-cells
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
        !(0 < ii < nx) && continue

        # the two halves of the control volume straddle the vertex `ox`, each half as wide
        # as the column it is cut from and as tall as the row it sits in
        half_dx = (@dx dxi ii) / 2
        min_corner = (ifelse(ii < i, ox - half_dx, ox), oy)
        p1 = GridGeometryUtils.Point(x[l], y[l])
        p2 = GridGeometryUtils.Point(x[l + 1], y[l + 1])
        chain_line = Line(p1, p2)
        y1 = line(chain_line, min_corner[1])
        y2 = line(chain_line, min_corner[1] + half_dx)
        p1 = GridGeometryUtils.Point(min_corner[1], y1)
        p2 = GridGeometryUtils.Point(min_corner[1] + half_dx, y2)
        s = Segment(p1, p2)

        ## create a rectangle for the new cell
        r = Rectangle(
            (min_corner[1] + half_dx / 2, min_corner[2] + dy / 2),
            half_dx, dy; θ = zero(half_dx)
        )
        # the halves differ in area on a refined grid, so their fractions are
        # area-weighted rather than averaged
        dA = half_dx * dy
        tmp += cell_rock_area(s, r) * dA
        ω += dA
    end
    ratios[i, j] = tmp / ω
end

function compute_area_below_chain_vy!(ratio_velocity, chain, xvi, dxi)
    topo_y = chain.h_vertices
    nx, ny = size(ratio_velocity)

    launch!(
        ka_backend(ratio_velocity), _compute_area_below_chain_vy!, (nx, ny),
        ratio_velocity, topo_y, xvi..., ny, dxi
    )
    return nothing
end

@kernel function _compute_area_below_chain_vy!(
        ratios::AbstractArray{T}, topo_y, xv, yv, ny, dxi
    ) where {T}
    I = @index(Global, NTuple)
    i, j = I

    dx = @dx dxi i
    ω = zero(T) # total area of the sampled sub-cells
    tmp = zero(T)
    # we can cache the potential coordinates
    x, y = (xv[i], xv[i + 1]), (topo_y[i], topo_y[i + 1])
    ox = xv[i]
    oy = yv[j]

    for jj in (j - 1):j
        !(0 < jj < ny) && continue

        # the two halves of the control volume straddle the vertex `oy`, each half as tall
        # as the row it is cut from and as wide as the column it sits in
        half_dy = (@dy dxi jj) / 2
        min_corner = (ox, ifelse(jj < j, oy - half_dy, oy))
        p1 = GridGeometryUtils.Point(x[1], y[1])
        p2 = GridGeometryUtils.Point(x[2], y[2])
        chain_line = Line(p1, p2)
        y1 = line(chain_line, min_corner[1])
        y2 = line(chain_line, min_corner[1] + dx)
        p1 = GridGeometryUtils.Point(min_corner[1], y1)
        p2 = GridGeometryUtils.Point(min_corner[1] + dx, y2)
        s = Segment(p1, p2)

        ## create a rectangle for the new cell
        r = Rectangle(
            (min_corner[1] + dx / 2, min_corner[2] + half_dy / 2),
            dx, half_dy; θ = zero(dx)
        )
        # the halves differ in area on a refined grid, so their fractions are
        # area-weighted rather than averaged
        dA = dx * half_dy
        tmp += cell_rock_area(s, r) * dA
        ω += dA
    end
    ratios[i, j] = tmp / ω
end

function compute_area_below_chain_vertex!(ratio_vertex, chain, xvi, dxi)
    topo_y = chain.h_vertices
    ni = size(ratio_vertex)

    launch!(
        ka_backend(ratio_vertex), _compute_area_below_chain_vertex!, ni,
        ratio_vertex, topo_y, xvi..., ni..., dxi
    )
    return nothing
end

@kernel function _compute_area_below_chain_vertex!(
        ratios::AbstractArray{T}, topo_y, xv, yv, nx, ny, dxi
    ) where {T}
    I = @index(Global, NTuple)
    i, j = I

    ω = zero(T) # total area of the sampled sub-cells
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
            !(0 < ii < nx) && continue
            !(0 < jj < ny) && continue

            # each quadrant of the control volume is half as wide and half as tall as the
            # cell it is cut from
            half_dx = (@dx dxi ii) / 2
            half_dy = (@dy dxi jj) / 2
            min_corner = (
                ifelse(ii < i, ox - half_dx, ox),
                ifelse(jj < j, oy - half_dy, oy),
            )
            p1 = GridGeometryUtils.Point(x[l], y[l])
            p2 = GridGeometryUtils.Point(x[l + 1], y[l + 1])
            chain_line = Line(p1, p2)
            y1 = line(chain_line, min_corner[1])
            y2 = line(chain_line, min_corner[1] + half_dx)
            p1 = GridGeometryUtils.Point(min_corner[1], y1)
            p2 = GridGeometryUtils.Point(min_corner[1] + half_dx, y2)
            s = Segment(p1, p2)

            ## create a rectangle for the new cell
            r = Rectangle(
                (min_corner[1] + half_dx / 2, min_corner[2] + half_dy / 2),
                half_dx, half_dy; θ = zero(half_dx)
            )
            # the quadrants differ in area on a refined grid, so their fractions are
            # area-weighted rather than averaged
            dA = half_dx * half_dy
            tmp += cell_rock_area(s, r) * dA
            ω += dA
        end
    end
    ratios[i, j] = tmp / ω
end

#############################

@inline function is_chain_above_cell(s::Segment, r::Rectangle)
    max_y = r.origin[2] + r.h / 2
    # Check if the segment is above the rectangle
    return GridGeometryUtils.geq_r(s.p1[2], max_y) && GridGeometryUtils.geq_r(s.p2[2], max_y)
end

@inline function is_chain_below_cell(s::Segment, r::Rectangle)
    min_y = r.origin[2] - r.h / 2
    # Check if the segment is below the rectangle
    return GridGeometryUtils.leq_r(s.p1[2], min_y) && GridGeometryUtils.leq_r(s.p2[2], min_y)
end

@inline function clip_chain_to_cell(s::Segment, r::Rectangle)
    min_y = r.origin[2] - r.h / 2
    max_y = r.origin[2] + r.h / 2
    dx = s.p2[1] - s.p1[1]
    dy = s.p2[2] - s.p1[2]

    y1 = clamp(s.p1[2], min_y, max_y)
    y2 = clamp(s.p2[2], min_y, max_y)
    x1 = s.p1[2] == y1 ? s.p1[1] : s.p1[1] + (y1 - s.p1[2]) * dx / dy
    x2 = s.p2[2] == y2 ? s.p2[1] : s.p1[1] + (y2 - s.p1[2]) * dx / dy

    return Segment(GridGeometryUtils.Point(x1, y1), GridGeometryUtils.Point(x2, y2))
end

# Chord and cell translated so that the cell is centered on the coordinate origin. Both the
# round-off tolerant comparisons above and `intersecting_area`'s check that the chord ends on
# the cell boundary size their tolerance relative to the magnitude of the coordinates they are
# handed, so in absolute coordinates a cell many cell widths from the origin falls inside its
# own tolerance: a `Float32` cell a few thousand widths out is swallowed whole.
@inline function recenter_on_cell(s::Segment, r::Rectangle{T}) where {T}
    ox, oy = r.origin[1], r.origin[2]
    s_local = Segment(
        GridGeometryUtils.Point(s.p1[1] - ox, s.p1[2] - oy),
        GridGeometryUtils.Point(s.p2[1] - ox, s.p2[2] - oy),
    )
    return s_local, Rectangle((zero(T), zero(T)), r.l, r.h; θ = zero(T))
end

"""
    cell_rock_area(s::Segment, r::Rectangle) -> Real

Fraction of the axis-aligned cell `r` lying below the marker chain segment `s`, in `[0, 1]`.

`s` spans the full width of `r`, runs left to right, and may leave the cell through its floor
or its ceiling.
"""
@inline function cell_rock_area(s::Segment, r::Rectangle{T}) where {T}
    s_local, cell = recenter_on_cell(s, r)

    is_chain_above_cell(s_local, cell) && return one(T)
    is_chain_below_cell(s_local, cell) && return zero(T)

    # A left-to-right chord has the rock region on its right-hand side.
    return clamp(
        intersecting_area(clip_chain_to_cell(s_local, cell), cell) / area(cell),
        zero(T), one(T)
    )
end
