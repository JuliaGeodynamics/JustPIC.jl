abstract type AbstractParticles end

"""
    Particles{Backend,N,I,T1,T2} <: AbstractParticles

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
struct Particles{Backend, N, I, T1, T2} <: AbstractParticles
    coords::NTuple{N, T1}
    index::T2
    nxcell::I
    max_xcell::I
    min_xcell::I
    np::I

    function Particles(
            ::Type{B},
            coords::NTuple{N, T1},
            index::T2,
            nxcell::I,
            max_xcell::I,
            min_xcell::I,
            np::I,
        ) where {B, N, I, T1, T2}
        return new{B, N, I, T1, T2}(coords, index, nxcell, max_xcell, min_xcell, np)
    end
end

function Particles(coords, index::CPUCellArray, nxcell, max_xcell, min_xcell, np)
    return Particles(CPUBackend, coords, index, nxcell, max_xcell, min_xcell, np)
end
struct MarkerChain{Backend, N, I, T1, T2, T3, TV} <: AbstractParticles
    coords::NTuple{N, T1}    # current x-coord in 2D, (x,y)
    coords0::NTuple{N, T1}   # x-coord in 2D, (x,y) from the previous time step
    h_vertices::T2          # topography at the vertices of the grid (current)
    h_vertices0::T2         # topography at the vertices of the grid (previous timestep)
    cell_vertices::TV
    index::T3
    max_xcell::I
    min_xcell::I

    function MarkerChain(
            ::Type{B},
            coords::NTuple{N, T1},
            coords0::NTuple{N, T1},
            h_vertices::T2,
            h_vertices0::T2,
            cell_vertices::TV,
            index::T3,
            min_xcell::I,
            max_xcell::I,
        ) where {B, N, I, T1, T2, T3, TV}
        return new{B, N, I, T1, T2, T3, TV}(
            coords,
            coords0,
            h_vertices,
            h_vertices0,
            cell_vertices,
            index,
            max_xcell,
            min_xcell,
        )
    end
end

function MarkerChain(coords, index::CPUCellArray, cell_vertices, min_xcell, max_xcell)
    return MarkerChain(
        CPUBackend,
        coords,
        coords0,
        h_vertices,
        h_vertices0,
        cell_vertices,
        index,
        max_xcell,
        min_xcell,
    )
end

struct PassiveMarkers{Backend, T} <: AbstractParticles
    coords::T
    np::Int64

    function PassiveMarkers(::Type{B}, coords::NTuple{N, T}) where {B, N, T}
        # np = length(coords[1].data)
        np = length(coords[1])
        return new{B, typeof(coords)}(coords, np)
    end
    function PassiveMarkers(::Type{B}, coords::AbstractArray) where {B}
        np = length(coords)
        return new{B, typeof(coords)}(coords, np)
    end
end

function PassiveMarkers(coords::Union{AbstractArray, NTuple{N, T}}) where {N, T}
    return PassiveMarkers(CPUBackend, coords)
end

# useful functions

unwrap_abstractarray(x::AbstractArray) = typeof(x).name.wrapper

@inline count_particles(p::AbstractParticles, icell::Vararg{Int, N}) where {N} =
    count(p.index[icell...])

@inline cell_length(p::MarkerChain{B, 2}) where {B} = p.cell_vertices[2] - p.cell_vertices[1]
@inline cell_length_x(p::MarkerChain{B, 3}) where {B} =
    p.cell_vertices[1][2] - p.cell_vertices[1][1]
@inline cell_length_y(p::MarkerChain{B, 3}) where {B} =
    p.cell_vertices[2][2] - p.cell_vertices[2][1]

@inline cell_x(p::AbstractParticles, icell::Vararg{Int, N}) where {N} = p.coords[1][icell...]
@inline cell_y(p::AbstractParticles, icell::Vararg{Int, N}) where {N} = p.coords[2][icell...]
@inline cell_z(p::AbstractParticles, icell::Vararg{Int, N}) where {N} = p.coords[3][icell...]

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

@inline function cell_index(x::NTuple{N, T}, xv::NTuple{N, AbstractVector{T}}) where {N, T}
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        cell_index(x[i], xv[i])
    end
end
