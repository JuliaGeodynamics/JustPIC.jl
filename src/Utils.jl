"""
    rangege(args...)

Make a linear range from `1` to `args[i]`, with `i ∈ [1, ..., n]`
"""
macro range(args...)
    return quote
        _range(tuple($(esc.(args)...))...)
    end
end

@inline Base.@pure _range(args::Vararg{Int,N}) where {N} = ntuple(i -> 1:args[i], Val(N))
@inline Base.@pure _range(args::NTuple{N,Int}) where {N} = ntuple(i -> 1:args[i], Val(N))

"""
    init_particle_fields(particles, ::Val{N})

Returns `N` particle fields with the same size as `particles`
"""
@inline Base.@pure function init_cell_arrays(particles, ::Val{N}) where N 
    return ntuple(_ -> @fill(0.0, size(particles.coords[1])..., celldims=(cellsize(particles.index))), Val(N))
end

@inline cell_array(x::T, ncells::NTuple{N, Integer}, ni::Vararg{Any, N}) where {T, N} = @fill(x, ni..., celldims=ncells, eltype=T) 
