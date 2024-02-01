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
    cell_vertices::NTuple{N,TV}
    max_xcell::I
    min_xcell::I

    function Particles(
        backend, coords::NTuple{N,T1}, index, max_xcell::I, min_xcell::I
    ) where {N,I,T1}

        # types
        T2 = typeof(index)
        # T3 = unwrap_abstractarray(coords[1].data)
        # permutations = T3(zeros(UInt8, size(index.data)...))

        return new{backend,N,max_xcell,I,T1,T2}(
            coords, index, cell_vertices, max_xcell, min_xcell
        )
    end
end

unwrap_abstractarray(x::AbstractArray) = typeof(x).name.wrapper

@inline count_particles(p::AbstractParticles, icell::Vararg{Int,N}) where {N} =
    count(p.index[icell...])

@inline cell_x(p::AbstractParticles, icell::Vararg{Int,N}) where {N} = p.coords[1][icell...]
@inline cell_y(p::AbstractParticles, icell::Vararg{Int,N}) where {N} = p.coords[2][icell...]
@inline cell_z(p::AbstractParticles, icell::Vararg{Int,N}) where {N} = p.coords[3][icell...]
