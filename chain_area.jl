
function compute_area_below_chain_centers!(ratio_center, chain, xvi, dxi)
    topo_x, topo_y = chain.cell_vertices, chain.h_vertices
    nx, ny = size(ratio_center)
    @parallel (1:nx, 1:ny) _compute_area_below_chain_center!(ratio_center, topo_x, topo_y, xvi..., dxi) 
    return nothing
end

@parallel_indices (i, j) function _compute_area_below_chain_center!(ratio, topo_x, topo_y, xv, yv, dxi) 
    _, dy = dxi

    # cell min max coordinates
    x_min_cell, x_max_cell = xv[i], xv[i + 1]
    y_min_cell, y_max_cell = yv[j], yv[j + 1]

    topo_xᵢ   = topo_x[i], topo_x[i + 1]
    topo_yᵢ   = topo_y[i], topo_y[i + 1]


    isbelow = topo_yᵢ[1] - y_min_cell > dy && topo_yᵢ[2] - y_min_cell > dy
    isabove = y_max_cell - topo_yᵢ[1] > dy && y_max_cell - topo_yᵢ[2] > dy

    if isbelow 
        ratio[i, j] = one(eltype(topo_xᵢ))

    elseif isabove
        ratio[i, j] = zero(eltype(topo_xᵢ))
    
    else
        # compute area at cell center
        area_rock   = compute_area_below_chain(
            topo_xᵢ, topo_yᵢ, x_min_cell, x_max_cell, y_min_cell, y_max_cell, dxi
        )
        area_cell   = prod(dxi)
        ratio[i, j] = clamp(area_rock / area_cell, 0, 1)
    end

    return nothing
end

function compute_area_below_chain(
    topo_xᵢ, topo_yᵢ, x_min_cell::T, x_max_cell::T, y_min_cell::T, y_max_cell::T, dxi
) where T <: Real

    dx, dy = dxi
    area = zero(T)

    is_chain_within_cell = true
    for y in topo_yᵢ
        is_chain_within_cell *= y_min_cell ≤ y ≤ y_max_cell
    end

    # whole topograpy segment is within the cell
    # := trapezoid area
    if is_chain_within_cell
        area += trapezoid_area((topo_yᵢ .- y_min_cell)..., dx)

    else
        # coordiantes of topography segment
        p1 = topo_xᵢ[1], topo_yᵢ[1]
        p2 = topo_xᵢ[2], topo_yᵢ[2]
        p  = p1, p2

        # compute intersections between topography segment and cell boundaries
        intersections = get_intersections(p, x_min_cell, x_max_cell, y_min_cell, y_max_cell)

        # compute inner area
        dx_intersections = intersections[2][1] - intersections[1][1]
        area += trapezoid_area(
            intersections[1][2] - y_min_cell, 
            intersections[2][2] - y_min_cell, 
            dx_intersections
        )

        # compute area of the tails, if necessary
        # compute left-hand-side tail
        if intersections[1][2] ≥ y_max_cell
            lx    = intersections[1][1] - x_min_cell
            area += rectangle_area(lx, dy)
        end
        # compute right-hand-side tail
        if intersections[2][2] ≥ y_max_cell
            lx    = x_max_cell - intersections[1][1]
            area += rectangle_area(lx, dy)
        end
    end

    return area 
end

### Utils

@inline trapezoid_area(a, b, h) = (a + b) * h / 2
@inline rectangle_area(lx, ly)  = lx * ly

function slope_intercept(p1, p2)
    # unpack
    x1, y1 = p1
    x2, y2 = p2
    # compute slope and intercept
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    return slope, intercept
end

function line_intersection(p1, p2, q1, q2)
    # compute slopes and intercepts
    a, c = slope_intercept(p1, p2)
    b, d = slope_intercept(q1, q2)
    # compute intersection point
    px = isinf(a) ? p1[1] : (d-c) / (a-b)
    py = @muladd b * px + d
    return (px, py)
end

function get_intersections(p::NTuple{2}, x_min_cell, x_max_cell, y_min_cell, y_max_cell)
    p1, p2 = p
    intersections = Base.@ntuple 2 ii -> begin
        @inline 
        pᵢ = p[ii]
        intersection = if pᵢ[2] ≥ y_max_cell
            line_intersection(
                p1,
                p2,
                (x_min_cell, y_max_cell), 
                (x_max_cell, y_max_cell), 
            )
        elseif pᵢ[2] ≤ y_min_cell
            line_intersection(
                p1,
                p2,
                (x_min_cell, y_min_cell), 
                (x_max_cell, y_min_cell), 
            )
        else
            pᵢ
        end
    end
    return intersections
end

function find_minmax_cell_indices(topo_yᵢ, origin_y, dy)
    ymin, ymax = extrema(topo_yᵢ)
    min_cell_j = Int((ymin - origin_y) ÷ dy) + 1
    max_cell_j = Int((ymax - origin_y) ÷ dy) + 1
    return min_cell_j, max_cell_j
end

# check if the topography segment intersects with the lateral cell boundaries;
# if it does, it does intersect with the cell
function do_intersect(p::NTuple{2}, x_min_cell, x_max_cell, y_min_cell, y_max_cell)
    p1, p2 = p
    
    _, left_y = line_intersection(
        p1,
        p2,
        (x_min_cell, y_min_cell), 
        (x_min_cell, y_max_cell), 
    )
    y_min_cell ≤ left_y ≤ y_max_cell && return true
    _, right_y = line_intersection(
        p1,
        p2,
        (x_max_cell, y_min_cell), 
        (x_max_cell, y_max_cell), 
    )
    y_min_cell ≤ right_y ≤ y_max_cell && return true

    return false
end