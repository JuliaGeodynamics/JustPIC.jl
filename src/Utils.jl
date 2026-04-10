"""
    add_global_ghost_nodes(x, dx, origin; backend = CPUBackend)

Add ghost coordinates only where the local coordinate vector touches the global
domain bounds.

`origin` is the global `(min, max)` extent in the same coordinate direction. If
`x` starts at `origin[1]`, a left ghost node is prepended; if it ends at
`origin[2]`, a right ghost node is appended. The result is converted with
`TA(backend)`.
"""
function add_global_ghost_nodes(x::AbstractArray, dx, origin; backend = CPUBackend)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    x1 == origin[1] && (x = vcat(xI, x))
    x2 == origin[2] && (x = vcat(x, xF))
    return x = TA(backend)(x)
end

"""
    add_ghost_nodes(x, dx, origin; backend = CPUBackend)

Return `x` with one ghost coordinate appended on both sides.

The `origin` argument is accepted for API symmetry with
`add_global_ghost_nodes` but is not used by this method. The result is converted
with `TA(backend)`.
"""
function add_ghost_nodes(x::AbstractArray, dx, origin; backend = CPUBackend)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    # LinRange(xI, xF, length(x)+2)
    return x = TA(backend)(vcat(xI, Array(x), xF))
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
    iszero(@inbounds @index index[ip, I...])

@generated function get_particle_coords(
        p::NTuple{N, CellArray}, ip, idx::Vararg{Int64, N}
    ) where {N}
    return quote
        @inline
        Base.@ntuple $N i -> @inbounds @index p[i][ip, idx...]
    end
end

function get_particle_coords(p::NTuple{N, CellArray}, ip, idx::Integer) where {N}
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        @inbounds @index p[i][ip, idx]
    end
end

function get_particle_coords(p::NTuple{N}, ip) where {N}
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        @inbounds p[i][ip]
    end
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
@inline find_parent_cell_bisection(px::Number, x::AbstractVector, seed) = find_parent_cell_bisection(px, x, 1, length(x), seed)

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
            lo = seed
            seed = div(hi + seed, 2)
        else
            hi = seed
            seed = div(lo + seed, 2)
        end
    end
    return
end
