"""
    PhaseRatios{Backend,T}

Storage for phase-fraction fields sampled at multiple grid locations.

Depending on dimension, the container holds phase ratios at cell centers,
vertices, staggered velocity nodes, and in 3D also at edge midpoints.

The fields store, for each location, the fractional occupancy of each material
phase inferred from particle labels.
"""
struct PhaseRatios{Backend, T} <: AbstractParticles
    center::T
    vertex::T
    Vx::T
    Vy::T
    Vz::T
    yz::T
    xz::T
    xy::T

    function PhaseRatios(
            ::Type{B}, center::T, vertex::T, Vx::T, Vy::T, Vz::T, yz::T, xz::T, xy::T
        ) where {B, T}
        return new{B, T}(center, vertex, Vx, Vy, Vz, yz, xz, xy)
    end
end

@inline dimension(::Type{PhaseRatios{Any, AbstractVector}}) = 1
@inline dimension(::Type{PhaseRatios{Any, AbstractMatrix}}) = 2
@inline dimension(::Type{PhaseRatios{Any, AbstractArray}}) = 3

"""
    nphases(x::PhaseRatios)

Return the number of phases in `x::PhaseRatios`.

This method returns a `Val` wrapper for the phase count; use `numphases` when
you need the integer directly.
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
