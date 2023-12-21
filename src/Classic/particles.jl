using StaticArrays

abstract type AsbtractParticles end

struct ClassicParticles{N,NP,T} <: AsbtractParticles
    coords::T
    grid_step::NTuple{N,Float64}
    ncells::NTuple{N,Int64}

    function ClassicParticles(nparticles, grid_lims, grid_step::NTuple{N,Float64}) where {N}
        ncells = Int64.(grid_lims ./ grid_step)
        coords = [@SVector(rand(N)) .* grid_lims for _ in 1:(nparticles * prod(ncells))]
        nparticles = length(coords)
        return new{N,nparticles,typeof(coords)}(coords, grid_step, ncells)
    end
end

# Cell indexing functions

@inline get_cell_i(x, p::ClassicParticles) = Int64(x ÷ p.grid_step[1] + 1)
@inline get_cell_j(y, p::ClassicParticles) = Int64(y ÷ p.grid_step[2] + 1)
@inline get_cell_k(z, p::ClassicParticles) = Int64(z ÷ p.grid_step[3] + 1)

function get_cell(xi, p::ClassicParticles{N}) where {N}
    ntuple(Val(N)) do i
        Base.@_inline_meta
        Int64(xi[i] ÷ p.grid_step[i] + 1)
    end
end

function get_cell(xi, dxi::NTuple{N,T}) where {N,T}
    ntuple(Val(N)) do i
        Base.@_inline_meta
        Int64(@inbounds xi[i] ÷ dxi[i] + 1)
    end
end

# Others

@inline nparticles(::ClassicParticles{N,NP}) where {N,NP} = NP
@inline dimension(::ClassicParticles{N}) where {N} = N
