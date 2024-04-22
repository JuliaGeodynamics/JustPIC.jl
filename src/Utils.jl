"""
    add_global_ghost_nodes(x, dx, origin)

Add ghost nodes to the global coordinates array `x` with spacing `dx` and origin `origin`    
"""
function add_global_ghost_nodes(x::AbstractArray, dx, origin)
    x1, x2 = extrema(x)
    xI = round(x1 - dx; sigdigits=5)
    xF = round(x2 + dx; sigdigits=5)
    x1 == origin[1] && (x = vcat(xI, x))
    x2 == origin[2] && (x = vcat(x, xF))
    return x = TA(x)
end

"""
    add_ghost_nodes(x, dx, origin)

Add ghost nodes to the local coordinates array `x` with spacing `dx` and origin `origin`    
"""
function add_ghost_nodes(x::AbstractArray, dx, origin)
    x1, x2 = extrema(x)
    xI = round(x1 - dx; sigdigits=5)
    xF = round(x2 + dx; sigdigits=5)
    return x = TA(vcat(xI, Array(x), xF))
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

@inline _idx(args::Vararg{Int,N}) where {N} = ntuple(i -> 1:args[i], Val(N))
@inline _idx(args::NTuple{N,Int}) where {N} = _idx(args...)

@inline doskip(index, ip, I::Vararg{Int64,N}) where {N} =
    iszero(@inbounds @cell index[ip, I...])

function get_particle_coords(p::NTuple{N,CellArray}, ip, idx::Vararg{Int64,N}) where {N}
    ntuple(Val(N)) do i
        Base.@_inline_meta
        @inbounds @cell p[i][ip, idx...]
    end
end

function get_particle_coords(p::NTuple{N,T}, ip) where {N,T}
    ntuple(Val(N)) do i
        Base.@_inline_meta
        p[i][ip]
    end
end
