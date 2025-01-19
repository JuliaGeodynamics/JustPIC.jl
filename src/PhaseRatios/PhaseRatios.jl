struct PhaseRatios{Backend, T} <: AbstractParticles
    center::T
    vertex::T
    Vx::T
    Vy::T
    Vz::T
    yz::T
    xz::T
    xy::T

    function PhaseRatios(::Type{B}, center::T, vertex::T, Vx::T, Vy::T, Vz::T, yz::T, xz::T, xy::T) where {B, T}
        return new{B, T}(center, vertex, Vx, Vy, Vz, yz, xz, xy)
    end
end

@inline dimension(::Type{PhaseRatios{Any, AbstractVector}}) = 1
@inline dimension(::Type{PhaseRatios{Any, AbstractMatrix}}) = 2
@inline dimension(::Type{PhaseRatios{Any, AbstractArray}}) = 3

"""
    nphases(x::PhaseRatios)

Return the number of phases in `x::PhaseRatios`.
"""
@inline nphases(x::PhaseRatios) = nphases(x.center)
@inline numphases(x::PhaseRatios) = numphases(x.center)

@inline function nphases(
        ::CellArray{StaticArraysCore.SArray{Tuple{N}, T, N1, N}}
    ) where {N, T, N1}
    return Val(N)
end

@inline function numphases(
        ::CellArray{StaticArraysCore.SArray{Tuple{N}, T, N1, N}}
    ) where {N, T, N1}
    return N
end
