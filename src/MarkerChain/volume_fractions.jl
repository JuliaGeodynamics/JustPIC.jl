using StaticArrays, MuladdMacro

## Area functions 

@inline cell_area(dx, dy) = dx * dy
@inline trapezoid_area(a, b) = trapezoid_area(a..., b...)
@inline trapezoid_area(x1, y1, x2, y2) = @muladd 0.5 * abs(x2 - x1) * (y1 + y2)

function area_below_local_chain(chain::MarkerChain, y_lower_left_corner::T, i::Int) where {T<:Real}

    h = y_lower_left_corner # height of the base of trapezoid
    x = @cell chain.coords[1][i]
    y = @cell chain.coords[2][i]
    # x0 = x_lower_left_corner

    A = zero(T)
    for k in 1:length(x)-1
        isnan(x[k+1]) && break
        A += @inbounds trapezoid_area(x[k], y[k] - h, x[k+1], y[k+1] - h)
    end

    # handle intersections with the wall
    ilast = find_last_particle(x)
    left_intersection, right_intersection = get_intersections(chain, i)
    # left hand side volume
    A += @inbounds trapezoid_area(left_intersection[1], left_intersection[2] - h, x[1], y[1] - h)
    # right hand side volume
    A += @inbounds trapezoid_area(x[ilast], y[ilast] - h, right_intersection[1], right_intersection[2] - h)
    
    return A
end

area_below_local_chain(chain, y_lower_left_corner, i)


# Compute the average height of the chain in the cell

function average_chain_height!(chain::MarkerChain)
    ncells = length(chain)
    @parallel (@idx ncells) average_chain_height_kernel!(chain::MarkerChain)
    return nothing
end

@parallel_indices (i) function average_chain_height_kernel!(chain::MarkerChain)
    chain.average_height[i] = average_chain_height(chain, i)
    return nothing
end

function average_chain_height(chain::MarkerChain, i::Int) 
    left_intersection, right_intersection = get_intersections(chain, i)
    v = left_intersection[2] + right_intersection[2]
    c = 2 # 0 + 2 because of the left and right intersections
    for ip in cellaxes(chain.coords[1])
        y = @index chain.coords[2][ip, i]
        isnan(y) && continue
        v += y
        c += 1
    end
    return v * inv(c)
end
@b average_chain_height($(chain, 1)...)

