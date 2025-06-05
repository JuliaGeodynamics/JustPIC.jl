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
    @parallel (1:nx, 1:ny) _compute_area_below_chain_center!(
        ratio_center, topo_y, xvi..., dxi
    )
    return nothing
end

@parallel_indices (i, j) function _compute_area_below_chain_center!(
        ratio::AbstractArray{T}, topo_y, xv, yv, dxi
    ) where {T}

    # cell origin
    ox = xv[i]
    oy = yv[j]

    p1 = GridGeometryUtils.Point(ox, topo_y[i])
    p2 = GridGeometryUtils.Point(xv[i+1], topo_y[i+1])
    s  = Segment(p1, p2)
    
    r = Rectangle((ox,oy), dxi...)
    ratio[i, j] = cell_rock_area(s, r)

    return nothing
end

function compute_area_below_chain_vx!(ratio_velocity, chain, xvi, dxi)
    topo_y = chain.h_vertices
    nx, ny = size(ratio_velocity)
    mask_x = (-0.5, 0e0) .* dxi[1]

    @parallel (1:nx, 1:ny) _compute_area_below_chain_vx!(
        ratio_velocity, topo_y, mask_x, xvi..., nx, dxi
    )
    return nothing
end

@parallel_indices (i, j) function _compute_area_below_chain_vx!(
        ratios::AbstractArray{T}, topo_y, mask_x, xv, yv, nx, dxi
    ) where {T}
    dx, dy = dxi
    half_dx = dx / 2
    half_dy = dy / 2
    c = 0
    ω = 0 # weight for the average
    tmp = zero(T)
    # we can cache the potential coordinates
    x, y =  if 1 < i < nx
        (xv[i-1], xv[i], xv[i+1]), (topo_y[i-1], topo_y[i], topo_y[i+1])
    elseif i == 1
        (xv[i], xv[i], xv[i+1]), (topo_y[i], topo_y[i], topo_y[i+1])
    else
        (xv[i-1], xv[i], xv[i]), (topo_y[i-1], topo_y[i], topo_y[i])
    end

    ox   = xv[i]
    oy   = yv[j]
    for (l, ii) in enumerate((i - 1):i)
        c += 1
        !(0 < ii < nx) && continue
        
        ω += 1

        ## new origin at the center of the (ii, jj)-th cell
        origin = (ox, oy) .+ (mask_x[c], zero(T))
        ## now we need to interpolate the segment of the chain to the boundaries of the new cell
        # segment of the chain
        p1 = GridGeometryUtils.Point(x[l], y[l])
        p2 = GridGeometryUtils.Point(x[l+1], y[l+1])
        # create a line from the two points
        l  = Line(p1, p2)
        # evaluate the line at the origin and origin + dx / 2
        y1 = line(l, origin[1])
        y2 = line(l, origin[1] + half_dx)
        # create two points at the boundaries of the new cell
        p1 = GridGeometryUtils.Point(origin[1], y1)
        p2 = GridGeometryUtils.Point(origin[1] + half_dx, y2)
        # and turn them into a segment
        s  = Segment(p1, p2)

        ## create a rectangle for the new cell
        r    = Rectangle(origin, half_dx, half_dy)
        tmp += cell_rock_area(s, r)
    end
    ratios[i,j] = tmp / ω
    return nothing
end

function compute_area_below_chain_vy!(ratio_velocity, chain, xvi, dxi)
    topo_y = chain.h_vertices
    nx, ny = size(ratio_velocity)
    mask_y = (-0.5, 0e0) .* dxi[2]

    @parallel (1:nx, 1:ny) _compute_area_below_chain_vy!(
        ratio_velocity, topo_y, mask_y, xvi..., ny, dxi
    )
    return nothing
end

@parallel_indices (i, j) function _compute_area_below_chain_vy!(
        ratios::AbstractArray{T}, topo_y, mask_y, xv, yv, ny, dxi
    ) where {T}
    dx, dy = dxi
    half_dx = dx / 2
    half_dy = dy / 2
    c = 0
    ω = 0 # weight for the average
    tmp = zero(T)
    # we can cache the potential coordinates
    x, y = (xv[i], xv[i+1]), (topo_y[i], topo_y[i+1])
    ox   = xv[i]
    oy   = yv[j]

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
        l  = Line(p1, p2)
        # evaluate the line at the origin and origin + dx / 2
        y1 = line(l, origin[1])
        y2 = line(l, origin[1] + half_dx)
        # create two points at the boundaries of the new cell
        p1 = GridGeometryUtils.Point(origin[1], y1)
        p2 = GridGeometryUtils.Point(origin[1] + half_dx, y2)
        # and turn them into a segment
        s  = Segment(p1, p2)

        ## create a rectangle for the new cell
        r    = Rectangle(origin, half_dx, half_dy)
        tmp += cell_rock_area(s, r)
    end
    ratios[i,j] = tmp / ω

    return nothing
end

function compute_area_below_chain_vertex!(ratio_vertex, chain, xvi, dxi)
    topo_y  = chain.h_vertices
    ni      = size(ratio_vertex)
    masks_x = (-0.5, 0e0, -0.5, 0e0) .* dxi[1]
    masks_y = (-0.5, -0.5, 0e0, 0e0) .* dxi[2]

    @parallel (@idx ni) _compute_area_below_chain_vertex!(
        ratio_vertex, topo_y, masks_x, masks_y, xvi..., ni..., dxi
    )
    return nothing
end

@parallel_indices (i, j) function _compute_area_below_chain_vertex!(
        ratios::AbstractArray{T}, topo_y, masks_x, masks_y, xv, yv, nx, ny, dxi
    ) where {T}
    dx, dy = dxi
    half_dx = dx / 2
    half_dy = dy / 2
    c = 0 # linear index of the mask
    ω = 0 # weight for the average
    tmp = zero(T)
    # we can cache the potential coordinates
    x, y =  if 1 < i < nx
        (xv[i-1], xv[i], xv[i+1]), (topo_y[i-1], topo_y[i], topo_y[i+1])
    elseif i == 1
        (xv[i], xv[i], xv[i+1]), (topo_y[i], topo_y[i], topo_y[i+1])
    else
        (xv[i-1], xv[i], xv[i]), (topo_y[i-1], topo_y[i], topo_y[i])
    end

    ox = xv[i]
    oy = yv[j]

    for (k, jj) in enumerate((j - 1):j)
        for (l, ii) in enumerate((i - 1):i)
            c += 1
            !(0 < jj < nx) && continue
            !(0 < ii < ny) && continue
            
            ω += 1

            ## new origin at the center of the (ii, jj)-th cell
            origin = (ox, oy) .+ (masks_x[c], masks_y[c])
            ## now we need to interpolate the segment of the chain to the boundaries of the new cell
            # segment of the chain
            p1 = GridGeometryUtils.Point(x[l], y[k])
            p2 = GridGeometryUtils.Point(x[l+1], y[k+1])
            # create a line from the two points
            l  = Line(p1, p2)
            # evaluate the line at the origin and origin + dx / 2
            y1 = line(l, origin[1])
            y2 = line(l, origin[1] + half_dx)
            # create two points at the boundaries of the new cell
            p1 = GridGeometryUtils.Point(origin[1], y1)
            p2 = GridGeometryUtils.Point(origin[1] + half_dx, y2)
            # and turn them into a segment
            s  = Segment(p1, p2)

            ## create a rectangle for the new cell
            r    = Rectangle(origin, half_dx, half_dy)
            tmp += cell_rock_area(s, r)
        end
    end
    ratios[i,j] = tmp / ω
    
    return nothing
end

#############################

@inline function is_chain_above_cell(s::Segment, r::Rectangle)
    max_y = r.origin[2] + r.h
    # Check if the segment is above the rectangle
    return s.p1[2] ≥ max_y && s.p2[2] ≥ max_y
end

@inline function is_chain_below_cell(s::Segment, r::Rectangle)
    min_y = r.origin[2]
    # Check if the segment is below the rectangle
    return s.p1[2] ≤ min_y && s.p2[2] ≤ min_y
end

function cell_rock_area(s::Segment, r::Rectangle{T}) where {T}
    A = if is_chain_above_cell(s, r)
        one(T)
    elseif is_chain_below_cell(s, r)
        zero(T)
    else
        clamp(intersecting_area(s, r) / area(r), zero(T), one(T))
    end
    return A 
end
