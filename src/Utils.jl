function add_global_ghost_nodes(x::AbstractArray, dx, origin; backend = CPUBackend)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    x1 == origin[1] && (x = vcat(xI, x))
    x2 == origin[2] && (x = vcat(x, xF))
    return x = TA(backend)(x)
end

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
