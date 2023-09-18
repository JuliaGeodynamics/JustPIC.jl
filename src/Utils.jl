"""
    range(args...)

Make a linear range from `1` to `args[i]`, with `i ∈ [1, ..., n]`
"""
macro range(args...)
    return quote
        _range(tuple($(esc.(args)...))...)
    end
end

@inline _range(args::Vararg{Int,N}) where {N} = ntuple(i -> 1:args[i], Val(N))
@inline _range(args::NTuple{N,Int}) where {N} = ntuple(i -> 1:args[i], Val(N))

"""
    init_particle_fields(particles, ::Val{N})

Returns `N` particle fields with the same size as `particles`
"""
@inline function init_cell_arrays(particles, ::Val{N}) where N 
    return ntuple(_ -> @fill(0.0, size(particles.coords[1])..., celldims=(cellsize(particles.index))), Val(N))
end

@inline cell_array(x::T, ncells::NTuple{N, Integer}, ni::Vararg{Any, N}) where {T, N} = @fill(x, ni..., celldims=ncells, eltype=T) 

"""
    add_global_ghost_nodes(x, dx, origin)

Add ghost nodes to the global coordinates array `x` with spacing `dx` and origin `origin`    
"""
function add_global_ghost_nodes(x::AbstractArray, dx, origin)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    x1 == origin[1] && (x = vcat(xI, x))
    x2 == origin[2] && (x = vcat(x, xF))
    x = TA(x)
end

"""
    add_ghost_nodes(x, dx, origin)

Add ghost nodes to the local coordinates array `x` with spacing `dx` and origin `origin`    
"""
function add_ghost_nodes(x::AbstractArray, dx, origin)
    x1, x2 = extrema(x)
    xI = round(x1-dx; sigdigits=5)
    xF = round(x2+dx; sigdigits=5)
    x = TA(vcat(xI, Array(x), xF))
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