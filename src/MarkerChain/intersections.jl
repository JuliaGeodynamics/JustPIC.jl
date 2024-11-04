using StaticArrays
import JustPIC._2D: @idx, _interp1D

## Methods to compute the intersection of the chain with the left and right hand side boundaries of their cells 

function compute_cell_intersections!(chain::MarkerChain)
    ncells = length(chain)
    @parallel (@idx ncells) compute_cell_intersections_kernel!(chain::MarkerChain, ncells)
    return nothing
end

@parallel_indices (i) function compute_cell_intersections_kernel!(chain::MarkerChain, ncells)
    
    # cache x-coords of the chain for the given cell
    x = @cell chain.coords[1][i]
    # Handle left-hand-side boundary
    if i == 1 # this is equivalent to iseven(ncells)

        ## 1. Compute left-hand-side intersection; extrapolate to the left wall
        xq = chain.cell_vertices[i] # need to interpolate to the left wall
        # particle on the lhs of the wall
        xl, yl = x[1], @index chain.coords[2][1, i]
        # particle on the rhs of the wall
        xr, yr = x[2], @index chain.coords[2][2, i]
        # interpolate to the left wall
        yq = _interp1D(xq, xr, xl, yr, yl)
        # intersection coordinates
        left_intersection = xq, yq
        ## 2. Compute right-hand-side intersection
        right_intersection = intersection_rhs(chain, x, i)

        # store intersections
        set_intersections!(chain, left_intersection, right_intersection, 1) 

    # Handle right-hand-side boundary
    elseif i == ncells # this is equivalent to iseven(ncells)
        ilast = find_last_particle(x)

        ## 1. Compute left-hand-side intersection; extrapolate to the left wall
        left_intersection  = intersection_lhs(chain, x, i)

        ## 2. Compute right-hand-side intersection
        xq = chain.cell_vertices[i+1] # need to interpolate to the left wall
        # particle on the rhs of the wall
        xl, yl = x[ilast-1], @index chain.coords[2][ilast - 1, i]
        # particle on the lhs of the wall
        xr, yr = x[ilast], @index chain.coords[2][ilast, i]
        # interpolate to the left wall
        yq = _interp1D(xq, xr, xl, yr, yl)
        # intersection coordinates
        right_intersection = xq, yq
        
        # store intersections
        set_intersections!(chain, left_intersection, right_intersection, ncells) 

    else
        ## 1. Compute left-hand-side intersection
        left_intersection  = intersection_lhs(chain, x, i)
        ## 2. Compute right-hand-side intersection
        right_intersection = intersection_rhs(chain, x, i)

        # store intersections
        set_intersections!(chain, left_intersection, right_intersection, i)  
    end

    return nothing
end

Base.@propagate_inbounds @inline function get_intersections(chain::MarkerChain, i::Int)
    left = (
        @index(chain.intersections[1, i]), 
        @index(chain.intersections[2, i]), 
    )
    right = (
        @index(chain.intersections[3, i]), 
        @index(chain.intersections[4, i]), 
    )
    return left, right
end

Base.@propagate_inbounds @inline function set_intersections!(chain::MarkerChain, left, right, i::Int)  
    for k in 1:2
        @index chain.intersections[k, i]     = left[k]
        @index chain.intersections[k + 2, i] = right[k]
    end
    return nothing
end

function find_last_particle(x)
    for i in length(x):-1:2 # start from the end
        isnan(x[i]) || return i
    end
    return 1
end

# Compute left-hand-side intersection
function intersection_lhs(chain::MarkerChain, x::SVector, i::Int)
    xq = chain.cell_vertices[i] # need to interpolate to the left wall
    # particle on the rhs of the wall; belongs to current cell
    xr, yr = x[1], @index chain.coords[2][1, i]
    # particle on the lhs of the wall; belongs to previous cell
    x_previous = @cell chain.coords[1][i-1]
    ilast_previous = find_last_particle(x)
    xl, yl = x_previous[ilast_previous], @index chain.coords[2][ilast_previous, i-1]
    # interpolate to the left wall
    yq = _interp1D(xq, xr, xl, yr, yl)
    # intersection coordinates
    left_intersection = xq, yq
    return left_intersection
end

function intersection_rhs(chain::MarkerChain, x::SVector, i::Int)
    ilast = find_last_particle(x)
    xq = chain.cell_vertices[i+1] # need to interpolate to the right wall 
    # particle on the lhs of the wall; belongs to current cell
    xl, yl = x[ilast], @index chain.coords[2][ilast, i]
    # particle on the rhs of the wall; belongs to next cell
    xr, yr = @index(chain.coords[1][1, i+1]), @index(chain.coords[2][1, i+1])
    # interpolate to the right wall
    yq = _interp1D(xq, xr, xl, yr, yl)
    # intersection coordinates
    right_intersection = xq, yq
    return right_intersection
end


