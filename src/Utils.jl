function add_global_ghost_nodes(x::AbstractArray, dx, origin; backend = CPU)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    x1 == origin[1] && (x = vcat(xI, x))
    x2 == origin[2] && (x = vcat(x, xF))
    return x = TA(backend)(x)
end

function add_ghost_nodes(x::AbstractArray, dx, origin; backend = CPU)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    # LinRange(xI, xF, length(x)+2)
    return x = TA(backend)(vcat(xI, Array(x), xF))
end

"""
    add_periodic_ghost_nodes(x::AbstractVector)

Extend a 1D periodic grid with one ghost node on each side.

The added coordinates preserve the spacing of the last and first physical cells,
respectively, which makes this helper work for both uniform and refined grids.

# Example
```julia
xv = [0.0, 0.25, 0.5, 0.75, 1.0]
xv_periodic = add_periodic_ghost_nodes(xv)
```
"""
function add_periodic_ghost_nodes(x::AbstractVector)
    length(x) ≥ 2 || throw(ArgumentError("At least two grid nodes are required"))

    dx_left = x[2] - x[1]
    dx_right = x[end] - x[end - 1]
    xI = x[1] - dx_right
    xF = x[end] + dx_left

    return vcat(xI, x, xF)
end

function add_periodic_ghost_nodes(x::LinRange)
    length(x) ≥ 2 || throw(ArgumentError("At least two grid nodes are required"))
    dx = step(x)
    return LinRange(first(x) - dx, last(x) + dx, length(x) + 2)
end

function add_periodic_ghost_nodes(x::AbstractRange)
    length(x) ≥ 2 || throw(ArgumentError("At least two grid nodes are required"))
    dx = step(x)
    return range(first(x) - dx; step = dx, length = length(x) + 2)
end

@inline function wrap_coordinate(x, periodic, limits)
    periodic || return x
    xmin, xmax = limits
    (xmin ≤ x < xmax) && return x
    return xmin + mod(x - xmin, xmax - xmin)
end

@inline function wrap_position(x::NTuple{N}, periodicity, domain_limits) where {N}
    return ntuple(i -> wrap_coordinate(x[i], periodicity[i], domain_limits[i]), Val(N))
end

"""
    @idx(args...)

Make a linear range from `1` to `args[i]`, with `i ∈ [1, ..., n]`
"""
macro idx(args...)
    return quote
        _idx(tuple($(esc.(args)...))...)
    end
end

@inline _idx(args::Vararg{Int, N}) where {N} = ntuple(i -> 1:args[i], Val(N))
@inline _idx(args::NTuple{N, Int}) where {N} = _idx(args...)

@inline doskip(index, ip, I::Vararg{Int64, N}) where {N} =
    iszero(@inbounds CAI.@index index[ip, I...])

@generated function get_particle_coords(
        p::NTuple{N, CellArray}, ip, idx::Vararg{Int64, N}
    ) where {N}
    return quote
        @inline
        Base.@ntuple $N i -> @inbounds CAI.@index p[i][ip, idx...]
    end
end

function get_particle_coords(p::NTuple{N, CellArray}, ip, idx::Integer) where {N}
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        @inbounds CAI.@index p[i][ip, idx]
    end
end

function get_particle_coords(p::NTuple{N}, ip) where {N}
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        @inbounds p[i][ip]
    end
end

@inline inner_size(A::AbstractArray) = size(A) .- 2
@inline function inner_ranges(A::AbstractArray{T, N}) where {T, N}
    return ntuple(i -> 1:(size(A, i) - 1), Val(N))
end

function inner_mask(::Particles{B, N}, ghosts::Vararg{Bool, 3}) where {B, N}
    return ntuple(i -> !(ghosts[i]) * - 1, Val(N))
end

###############################
# MACROS TO INDEX GRID ARRAYS #
###############################

macro dxi(args...)
    return :(_dxi($(esc.(args)...)))
end

Base.@propagate_inbounds @inline _dxi(dxi::NTuple{2, Union{Number, AbstractVector}}, I::Integer, J::Integer) = _dx(dxi, I), _dy(dxi, J)
Base.@propagate_inbounds @inline _dxi(dxi::NTuple{3, Union{Number, AbstractVector}}, I::Integer, J::Integer, K::Integer) = _dx(dxi, I), _dy(dxi, J), _dz(dxi, K)

macro dx(args...)
    return :(_dx($(esc.(args)...)))
end

Base.@propagate_inbounds @inline _dx(dx::NTuple{N, Union{Number, AbstractVector}}, I::Integer) where {N} = getindex_dxi(dx[1], I)

macro dy(args...)
    return :(_dy($(esc.(args)...)))
end

Base.@propagate_inbounds @inline _dy(dy::NTuple{N, Union{Number, AbstractVector}}, I::Integer) where {N} = getindex_dxi(dy[2], I)

macro dz(args...)
    return :(_dz($(esc.(args)...)))
end

Base.@propagate_inbounds @inline _dz(dz::NTuple{3, Union{Number, AbstractVector}}, I::Integer) = getindex_dxi(dz[3], I)

Base.@propagate_inbounds @inline getindex_dxi(dxi::AbstractVector, I::Integer) = dxi[I]
Base.@propagate_inbounds @inline getindex_dxi(dxi::Number, ::Integer) = dxi

#######################
# BISECTION ALGORITHM #
#######################

"""
    find_parent_cell_bisection(px::Number, x::AbstractVector, seed::Int)

Performs an iterative bisection search on the cell-edge vector `x` to find the index of the cell containing `px`,
starting from the initial guess `seed`.

# Arguments
- `px::Number`: Coordinate of the point we want to locate.
- `x::AbstractVector`: Monotonic vector of cell-edge coordinates.
- `seed::Int`: Initial cell index guess used to start the search.

# Returns
- An integer index `i` such that `x[i] ≤ px ≤ x[i + 1]`.
"""
@inline find_parent_cell_bisection(px::Number, x::AbstractVector, seed) =
    find_parent_cell_bisection(px, x, 1, length(x) - 1, clamp(seed, 1, length(x) - 1))

@generated function find_parent_cell_bisection(px::NTuple{N, Number}, x::NTuple{N, AbstractVector}, seed) where {N}
    return quote
        @inline
        Base.@ntuple $N i -> find_parent_cell_bisection(px[i], x[i], seed[i])
    end
end

@inline function find_parent_cell_bisection(px, x, lo, hi, seed)
    while true
        x[seed] ≤ px ≤ x[seed + 1] && return seed

        if x[seed] < px
            lo = seed + 1
        else
            hi = seed - 1
        end
        lo > hi && return clamp(seed, 1, length(x) - 1)
        seed = div(lo + hi, 2)
    end
end

"""
    parent_cell_index(x, xv, seed)

Return the index `i` of the cell of the vertex vector `xv` that contains `x`, i.e. the `i`
such that `xv[i] ≤ x < xv[i + 1]`, clamped to `1:length(xv) - 1`.

`xv` may be uniformly spaced (an `AbstractRange`, resolved arithmetically) or refined (any
other `AbstractVector`, resolved by bisection from the initial guess `seed`). `seed` is
ignored in the uniform case.
"""
@inline parent_cell_index(x, xv::AbstractRange, seed) =
    clamp(cell_index(x, xv, step(xv)), 1, length(xv) - 1)
@inline function parent_cell_index(x, xv::AbstractVector, seed)
    i = find_parent_cell_bisection(x, xv, seed)
    # the bisection brackets inclusively at both ends; cells are half-open
    # `[xv[i], xv[i + 1])`, so a point sitting exactly on an interior vertex
    # belongs to the cell on its right, as it does on a range
    return ifelse(i < length(xv) - 1 && x == xv[i + 1], i + 1, i)
end

# Seed for a lookup with no caller context: starting from the middle cell bounds the number
# of bisection steps to log2 of the cell count.
@inline midpoint_seed(xv::AbstractVector) = max(length(xv) >> 1, 1)

"""
    cell_width(xv, i)

Width of cell `i` of the vertex vector `xv`.
"""
@inline cell_width(xv, i::Integer) = xv[i + 1] - xv[i]
