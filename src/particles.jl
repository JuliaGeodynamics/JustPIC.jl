abstract type AbstractParticles end

"""
    struct Particles{Backend,N,M,I,T1,T2} <: AbstractParticles

A struct representing a collection of particles.

# Arguments
- `backend`: The backend used for particle computations (CPUBackend, CUDABackend, AMDGPUBackend).
- `coords`: Coordinates of the particles
- `index`: Helper array flaggin active particles
- `nxcell`: Initial number of particles per cell
- `max_xcell`: Maximum number of particles per cell
- `min_xcell`: Minimum number of particles per cell
- `np`: Number of particles
"""
struct Particles{Backend,N,M,I,T1,T2} <: AbstractParticles
    coords::NTuple{N,T1}
    index::T2
    nxcell::I
    max_xcell::I
    min_xcell::I
    np::I

    function Particles(
        backend,
        coords::NTuple{N,T1},
        index::T2,
        nxcell::I,
        max_xcell::I,
        min_xcell::I,
        np::I,
    ) where {N,I,T1,T2}
        return new{backend,N,max_xcell,I,T1,T2}(
            coords, index, nxcell, max_xcell, min_xcell, np
        )
    end
end

Particles(coords, index::CPUCellArray, nxcell, max_xcell, min_xcell, np) =
    Particles(CPUBackend, coords, index, nxcell, max_xcell, min_xcell, np)

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

MarkerChain(coords, index::CPUCellArray, cell_vertices, min_xcell, max_xcell) =
    MarkerChain(CPUBackend, coords, index, cell_vertices, min_xcell, max_xcell)

struct PassiveMarkers{Backend,T} <: AbstractParticles
    coords::T
    np::Int64

    function PassiveMarkers(backend, coords::NTuple{N,T}) where {N,T}
        # np = length(coords[1].data)
        np = length(coords[1])
        return new{backend,typeof(coords)}(coords, np)
    end
    function PassiveMarkers(backend, coords::AbstractArray)
        np = length(coords)
        return new{backend,typeof(coords)}(coords, np)
    end
end

PassiveMarkers(coords::Union{AbstractArray, NTuple{N,T}}) where {N,T} = PassiveMarkers(CPUBackend, coords)

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

# @inline cell_index(xᵢ::T, dxᵢ::T) where {T} = abs(Int(xᵢ ÷ dxᵢ)) + 1
@inline cell_index(xᵢ::T, dxᵢ::T) where {T} = abs(Int(trunc(xᵢ / dxᵢ))) + 1
@inline cell_index(xᵢ::T, xvᵢ::AbstractVector{T}) where {T} =
    cell_index(xᵢ, xvᵢ, abs(xvᵢ[2] - xvᵢ[1]))

# generic one that works for any grid
@inline function cell_index(xᵢ::T, xvᵢ::AbstractVector{T}, dxᵢ::T) where {T}
    xv₀ = first(xvᵢ)
    if iszero(xv₀)
        return cell_index(xᵢ, dxᵢ)
    else
        return cell_index(xᵢ - xv₀, dxᵢ)
    end
end

@inline function cell_index(x::NTuple{N,T}, xv::NTuple{N,AbstractVector{T}}) where {N,T}
    ntuple(Val(N)) do i
        Base.@_inline_meta
        cell_index(x[i], xv[i])
    end
end
