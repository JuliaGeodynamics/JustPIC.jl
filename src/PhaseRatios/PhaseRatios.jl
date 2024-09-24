struct PhaseRatios{Backend,T}
    center::T
    vertex::T
end

"""
    nphases(x::PhaseRatios)

Return the number of phases in `x::PhaseRatios`.
"""
@inline nphases(x::PhaseRatios) = nphases(x.center)
@inline numphases(x::PhaseRatios) = numphases(x.center)

@inline function nphases(
    ::CellArray{StaticArraysCore.SArray{Tuple{N},T,N1,N}}
) where {N,T,N1}
    return Val(N)
end

@inline function numphases(
    ::CellArray{StaticArraysCore.SArray{Tuple{N},T,N1,N}}
) where {N,T,N1}
    return N
end