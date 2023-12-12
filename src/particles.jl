abstract type AsbtractParticles end

struct Particles{N,M,I,T1,T2,T3}  <: AsbtractParticles
    coords::NTuple{N,T1}
    index::T2
    inject::T3
    move::T2
    nxcell::I
    max_xcell::I
    min_xcell::I
    np::I

    function Particles(
        coords::NTuple{N,T1},
        index,
        inject,
        move,
        nxcell::I,
        max_xcell::I,
        min_xcell::I,
        np::I,
    ) where {N,I,T1}

        # types
        T2 = typeof(index)
        T3 = typeof(inject)
        return new{N,max_xcell,I,T1,T2,T3}(
            coords, index, inject, move, nxcell, max_xcell, min_xcell, np
        )
    end
end

struct ParticlesCloud{N, NP, T1, T2} <: AsbtractParticles
    coords::T1
    parent_cell::T2
    grid_step::NTuple{N, Float64}

    function ParticlesCloud(coords::T1, parent_cell::T2, nparticles, grid_step::NTuple{N, Float64}) where {N, T1, T2}
        new{N, nparticles, T1, T2}(coords, parent_cell, grid_step)
    end
end

struct MarkerChain{N,I,T1,T2}  <: AsbtractParticles
    coords::NTuple{N,T1}
    index::T2
    nxcell::I
    max_xcell::I
    min_xcell::I

    function MarkerChain(
        coords::NTuple{N,T1},
        index,
        nxcell::I,
        max_xcell::I,
        min_xcell::I,
    ) where {N,I,T1}

        # types
        T2 = typeof(index)
        return new{N,I,T1,T2}(
            coords, index, nxcell, max_xcell, min_xcell
        )
    end
end

# Sorting 

# Cell indexing functions

@inline get_cell_i(x, p::ParticlesCloud) = Int64(x ÷ p.grid_step[1] + 1) 
@inline get_cell_j(y, p::ParticlesCloud) = Int64(y ÷ p.grid_step[2] + 1) 
@inline get_cell_k(z, p::ParticlesCloud) = Int64(z ÷ p.grid_step[3] + 1) 

@inline get_cell_i(x::T, dx::T) where T<:Real = Int64(x ÷ dx + 1) 
@inline get_cell_j(y::T, dy::T) where T<:Real = Int64(y ÷ dy + 1) 
@inline get_cell_k(z::T, dz::T) where T<:Real = Int64(z ÷ dz + 1) 

function get_cell(xi, p::ParticlesCloud{N}) where N 
    ntuple(Val(N)) do i 
        Base.@_inline_meta
        Int64(xi[i] ÷ p.grid_step[i] + 1) 
    end
end

function get_cell(xi::Union{SVector{N, T}, NTuple{N, T}}, dxi::NTuple{N, T}) where {N, T<:Real}
    ntuple(Val(N)) do i 
        Base.@_inline_meta
        Int64(xi[i] ÷ dxi[i] + 1) 
    end
end

# Sorting 


# Others

@inline nparticles(::ParticlesCloud{N, NP}) where {N, NP} = NP
@inline dimension(::ParticlesCloud{N}) where {N} = N