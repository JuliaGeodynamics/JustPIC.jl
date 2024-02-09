abstract type AbstractParticles end

struct Particles{Backend,N,M,I,T1,T2,T3} <: AbstractParticles
    coords::NTuple{N,T1}
    index::T2
    inject::T3
    nxcell::I
    max_xcell::I
    min_xcell::I
    np::I

    function Particles(
        backend,
        coords::NTuple{N,T1},
        index,
        inject,
        nxcell::I,
        max_xcell::I,
        min_xcell::I,
        np::I,
    ) where {N,I,T1}

        # types
        T2 = typeof(index)
        T3 = typeof(inject)

        return new{backend,N,max_xcell,I,T1,T2,T3}(
            coords, index, inject, nxcell, max_xcell, min_xcell, np
        )
    end
end

struct MarkerChain{Backend,N,M,I,T1,T2,TV} <: AbstractParticles
    coords::NTuple{N,T1}
    index::T2
    cell_vertices::TV # x-coord in 2D, (x,y)-coords in 3D
    max_xcell::I
    min_xcell::I

    function MarkerChain(
        backend,
        coords::NTuple{N,T1},
        index::T2,
        cell_vertices::TV,
        min_xcell::I,
        max_xcell::I,
    ) where {N,I,T1,T2,TV}
        return new{backend,N,max_xcell,I,T1,T2,TV}(
            coords, index, cell_vertices, max_xcell, min_xcell
        )
    end
end

struct PassiveMarkers{Backend,N,T} <: AbstractParticles
    coords::NTuple{N,T}
    np::Int64

    function PassiveMarkers(backend, coords::NTuple{N,T}) where {N,T}
        np = length(coords[1].data)
        return new{backend,N,T}(coords, np)
    end
end


# useful functions

unwrap_abstractarray(x::AbstractArray) = typeof(x).name.wrapper

@inline count_particles(p::AbstractParticles, icell::Vararg{Int,N}) where {N} =
    count(p.index[icell...])

@inline cell_length(p::MarkerChain{B,2}) where {B} = p.cell_vertices[2] - p.cell_vertices[1]
@inline cell_length_x(p::MarkerChain{B,3}) where {B} =
    p.cell_vertices[1][2] - p.cell_vertices[1][1]
@inline cell_length_y(p::MarkerChain{B,3}) where {B} =
    p.cell_vertices[2][2] - p.cell_vertices[2][1]

@inline cell_x(p::AbstractParticles, icell::Vararg{Int,N}) where {N} = p.coords[1][icell...]
@inline cell_y(p::AbstractParticles, icell::Vararg{Int,N}) where {N} = p.coords[2][icell...]
@inline cell_z(p::AbstractParticles, icell::Vararg{Int,N}) where {N} = p.coords[3][icell...]

@inline cell_index(xᵢ::T, dxᵢ::T) where {T} = abs(Int(xᵢ ÷ dxᵢ)) + 1
@inline cell_index(xᵢ::T, xvᵢ::AbstractRange{T}) where {T} =
    cell_index(xᵢ, xvᵢ, xvᵢ[2] - xvᵢ[1])

# generic one that works for any grid
@inline function cell_index(xᵢ::T, xvᵢ::AbstractRange{T}, dxᵢ::T) where {T}
    xv₀ = first(xvᵢ)
    return iszero(xv₀) ? cell_index(xᵢ, dxᵢ) : cell_index(xᵢ - xv₀, dxᵢ)
end

@inline function cell_index(x::NTuple{N,T}, xv::NTuple{N,AbstractRange{T}}) where {N,T}
    ntuple(Val(N)) do i
        Base.@_inline_meta
        cell_index(x[i], xv[i])
    end
end
