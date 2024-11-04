struct PhaseRatios{Backend,T} <: AbstractParticles
    center::T
    vertex::T
    Vx::T
    Vy::T
    Vz::T
    yz::T
    xz::T
    xy::T

    function PhaseRatios(::Type{B}, center::T, vertex::T, Vx::T, Vy::T, Vz::T, yz::T, xz::T, xy::T) where {B, T}
        return new{B, T}(center, vertex)
    end
end

Adapt.@adapt_structure PhaseRatios


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