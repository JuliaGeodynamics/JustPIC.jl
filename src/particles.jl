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
    cell_vertices::TV # x-coord in 1D, (x,y)-coords in 2D
    max_xcell::I
    min_xcell::I

    function MarkerChain(
        backend, coords::NTuple{N,T1}, index, cell_vertices::TV, max_xcell::I, min_xcell::I
    ) where {N,I,T1,TV}

        # types
        T2 = typeof(index)
        # T3 = unwrap_abstractarray(coords[1].data)
        # permutations = T3(zeros(UInt8, size(index.data)...))

        return new{backend,N,max_xcell,I,T1,T2,TV}(
            coords, index, cell_vertices, max_xcell, min_xcell
        )
    end
end

unwrap_abstractarray(x::AbstractArray) = typeof(x).name.wrapper

@inline count_particles(p::AbstractParticles, icell::Vararg{Int,N}) where {N} =
    count(p.index[icell...])

@inline cell_length(p::MarkerChain{B, 2})   where B = p.cell_vertices[2] - p.cell_vertices[1]
@inline cell_length_x(p::MarkerChain{B, 3}) where B = p.cell_vertices[1][2] - p.cell_vertices[1][1]
@inline cell_length_y(p::MarkerChain{B, 3}) where B = p.cell_vertices[2][2] - p.cell_vertices[2][1]

@inline cell_x(p::AbstractParticles, icell::Vararg{Int,N})       where {N}    = p.coords[1][icell...]
@inline cell_y(p::AbstractParticles, icell::Vararg{Int,N})       where {N}    = p.coords[2][icell...]
@inline cell_z(p::AbstractParticles{B, 3}, icell::Vararg{Int,N}) where {B, N} = p.coords[3][icell...]

