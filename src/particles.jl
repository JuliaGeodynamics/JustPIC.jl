abstract type AsbtractParticles end

struct Particles{N,M,I,T1,T2,T3} <: AsbtractParticles
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

struct ParticlesCloud{N,NP,T1,T2} <: AsbtractParticles
    coords::T1
    parent_cell::T2
    grid_step::NTuple{N,Float64}

    function ParticlesCloud(
        coords::T1, parent_cell::T2, nparticles, grid_step::NTuple{N,Float64}
    ) where {N,T1,T2}
        return new{N,nparticles,T1,T2}(coords, parent_cell, grid_step)
    end
end

struct MarkerChain{N,I,T1,T2} <: AsbtractParticles
    coords::NTuple{N,T1}
    index::T2
    nxcell::I
    max_xcell::I
    min_xcell::I

    function MarkerChain(
        coords::NTuple{N,T1}, index, nxcell::I, max_xcell::I, min_xcell::I
    ) where {N,I,T1}

        # types
        T2 = typeof(index)
        return new{N,I,T1,T2}(coords, index, nxcell, max_xcell, min_xcell)
    end
end

# Sorting 

# Cell indexing functions

@inline get_cell_i(x, p::ParticlesCloud) = Int64(x ÷ p.grid_step[1] + 1)
@inline get_cell_j(y, p::ParticlesCloud) = Int64(y ÷ p.grid_step[2] + 1)
@inline get_cell_k(z, p::ParticlesCloud) = Int64(z ÷ p.grid_step[3] + 1)

@inline get_cell_i(x::T, dx::T) where {T<:Real} = Int64(x ÷ dx + 1)
@inline get_cell_j(y::T, dy::T) where {T<:Real} = Int64(y ÷ dy + 1)
@inline get_cell_k(z::T, dz::T) where {T<:Real} = Int64(z ÷ dz + 1)

function get_cell(xi, p::ParticlesCloud{N}) where {N}
    ntuple(Val(N)) do i
        Base.@_inline_meta
        Int64(xi[i] ÷ p.grid_step[i] + 1)
    end
end

function get_cell(xi::Union{SVector{N,T},NTuple{N,T}}, dxi::NTuple{N,T}) where {N,T<:Real}
    ntuple(Val(N)) do i
        Base.@_inline_meta
        Int64(xi[i] ÷ dxi[i] + 1)
    end
end

# Others

@inline nparticles(::ParticlesCloud{N,NP}) where {N,NP} = NP
@inline dimension(::ParticlesCloud{N}) where {N} = N

# Initialization

function init_particles(
    nxcell, max_xcell, min_xcell, xi, di, ni::NTuple{N,Integer}
) where {N}
    ncells = prod(ni)
    np = max_xcell * ncells
    # allocate necesary arrays for the particles struct
    p_xi = ntuple(_ -> @rand(ni..., celldims = (max_xcell,)), Val(N))
    inject = @fill(false, ni..., eltype = Bool)
    index = @fill(false, ni..., celldims = (max_xcell,), eltype = Bool)
    move = @fill(false, ni..., celldims = (max_xcell,), eltype = Bool)

    @parallel_indices (I...) function fill_coords_index(
        p_xi, index, xi, di::NTuple{N,T}, nxcell, max_xcell
    ) where {N,T}
        # lower-left corner of the cell
        x0i = ntuple(k -> xi[I[k]], Val(N))
        # fill index array
        for l in 1:max_xcell
            if l ≤ nxcell
                ntuple(Val(N)) do k
                    Base.@_inline_meta
                    @cell p_xi[k][l, i, j] =
                        x0i[k] + di[k] * (@cell(p_xi[k][l, I...]) * 0.9 + 0.05)
                    nothing
                end
                @cell index[l, I...] = true

            else
                ntuple(Val(N)) do k
                    Base.@_inline_meta
                    @cell p_xi[k][l, i, j] = NaN
                    nothing
                end
            end
        end
        return nothing
    end

    @parallel (@idx ni) fill_coords_index(p_xi..., index, xi..., di..., nxcell, max_xcell)

    return Particles(p_xi, index, inject, move, nxcell, max_xcell, min_xcell, np)
end

## Cloud of particles 

function init_particles_cloud(nxcell, xi::NTuple{N,T}, di, ni) where {N,T}
    ncells = prod(ni)
    np = nxcell * ncells

    @inline random_location(di) = (rand() * 0.9 + 0.05) * di

    pcoords = TA([SVector{N,Float64}(random_location(di[i]) for i in 1:N) for _ in 1:np])
    parent_cell = TA([ntuple(_ -> zero(T), Val(N)) for _ in 1:np])

    ni2 = ntuple(i -> length(xi[i]) - 1, Val(N))
    LinInd = LinearIndices(@idx ni2)

    @parallel_indices (I...) function fill_coords_index(
        pcoords, parent_cell, xi::NTuple{N,T}, nxcell, LinInd
    ) where {N,T}
        k = LinInd[I...]
        # lower-left corner of the cell
        x0i = ntuple(i -> xi[i], Val(N))
        # fill index array
        for l in (1 + nxcell * (k - 1)):(nxcell * k)
            pcoords[l] = SVector{N,T}(x0i[m] + pcoords[l][m] for m in 1:N)
            parent_cell[l] = I
        end
        return nothing
    end

    @parallel (@idx ni) fill_coords_index(pcoords, parent_cell, xi, nxcell, LinInd)

    return ParticlesCloud(pcoords, parent_cell, np, (dx, dy))
end
