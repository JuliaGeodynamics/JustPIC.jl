## CLASSIC PIC ------------------------------------------------------------------------------------------------

# LAUNCHERS
function grid2particle!(Fp, xvi, F, particles)
    (; coords, index) = particles
    di = grid_size(xvi)
    ni = size(index) .- 2
    @parallel (@idx ni) grid2particle_classic!(Fp, F, xvi, index, di, coords)

    return nothing
end

@parallel_indices (I...) function grid2particle_classic!(
        Fp, F, xvi, index, di, particle_coords
    )
    _grid2particle_classic!(
        Fp, particle_coords, xvi, di, F, index, tuple(I...), Val(cellnum(index))
    )
    return nothing
end

# INNERMOST INTERPOLATION KERNEL

@inline function _grid2particle_classic!(
        Fp::AbstractArray, p, xvi, di::NTuple{2}, F::AbstractArray, index, idx
    )
    i, j, ip = idx
    iCA, jCA = i + 1, j + 1
    # iterate over all the particles within the cells of index `idx`
    # skip lines below if there is no particle in this piece of memory
    return if !doskip(index, ip, iCA, jCA)
        Fi = field_corners(F, (i, j))
        # cache particle coordinates
        pᵢ = get_particle_coords(p, ip, iCA, jCA)
        # Interpolate field F onto particle
        @index Fp[ip, iCA, jCA] = _grid2particle(pᵢ, xvi, di, Fi, (i, j))
    end
end

@generated function _grid2particle_classic!(
        Fp, p, xvi, di, F, index, idx, ::Val{N}
    ) where {N}
    return quote
        Base.@_inline_meta
        idxCA = idx .+ 1
        Fi = field_corners(F, idx)
        xi_corner = corner_coordinate(xvi, idx)
        # iterate over all the particles within the cells of index `idx`
        Base.@nexprs $N ip -> begin
            # skip lines below if there is no particle in this piece of memory
            @inbounds if !doskip(index, ip, idxCA...)
                # cache particle coordinates
                pᵢ = get_particle_coords(p, ip, idxCA...)
                # Interpolate field F onto particle
                @index Fp[ip, idxCA...] = _grid2particle(pᵢ, xi_corner, di, Fi)
            end
        end
        return nothing
    end
end

@inline function _grid2particle_classic!(
        Fp::NTuple{N1, T1}, p, xvi, di::NTuple{N2, T2}, F::NTuple{N1, T3}, index, idx
    ) where {N1, T1, N2, T2, T3}
    idxCA = idx .+ 1
    # iterate over all the particles within the cells of index `idx`
    return @inbounds for ip in cellaxes(Fp[1])
        # skip lines below if there is no particle in this piece of memory
        doskip(index, ip, idxCA...) && continue

        # cache particle coordinates
        pᵢ = get_particle_coords(p, ip, idxCA...)

        # skip lines below if there is no particle in this piece of memory
        any(isnan, pᵢ) && continue

        # Interpolate field F onto particle
        ntuple(Val(N1)) do i
            Base.@_inline_meta
            @index Fp[i][ip, idxCA...] = _grid2particle(pᵢ, xvi, di, F[i], idx)
        end
        return nothing
    end
end

## FULL PARTICLE PIC ------------------------------------------------------------------------------------------

# LAUNCHERS

function grid2particle_flip!(Fp, xvi, F, F0, particles; α = 0.0)
    di = grid_size(xvi)
    (; coords, index) = particles
    ni = size(index) .- 2
    @parallel (@idx ni) grid2particle_full!(Fp, F, F0, xvi, di, coords, index, α)

    return nothing
end

@parallel_indices (I...) function grid2particle_full!(
        Fp, F, F0, xvi, di, particle_coords, index, α
    )
    _grid2particle_full!(Fp, particle_coords, xvi, di, F, F0, index, I, α)
    return nothing
end

# INNERMOST INTERPOLATION KERNEL

@inline function _grid2particle_full!(
        Fp, p, xvi, di::NTuple{N, T}, F, F0, index, idx, α
    ) where {N, T}
    idxCA = idx .+ 1
    Fi = field_corners(F, idx)
    F0i = field_corners(F0, idx)

    # iterate over all the particles within the cells of index `idx`
    @inbounds for ip in cellaxes(Fp)
        # skip lines below if there is no particle in this piece of memory
        doskip(index, ip, idxCA...) && continue

        # cache particle coordinates
        pᵢ = get_particle_coords(p, ip, idxCA...)

        # # skip lines below if there is no particle in this piece of memory
        # any(isnan, pᵢ) && continue

        Fᵢ = @index Fp[ip, idxCA...]
        F_pic, F0_pic = _grid2particle(pᵢ, xvi, di, (Fi, F0i), idx)
        ΔF = F_pic - F0_pic
        F_flip = Fᵢ + ΔF
        # Interpolate field F onto particle
        @index Fp[ip, idxCA...] = muladd(F_pic, α, F_flip * (1 - α))
    end
    return nothing
end

@inline function _grid2particle_full!(
        Fp::NTuple{N1, T1},
        p,
        xvi,
        di::NTuple{N2, T2},
        F::NTuple{N1, T3},
        F0::NTuple{N1, T3},
        index,
        idx,
        α,
    ) where {N1, T1, N2, T2, T3}
    idxCA = idx .+ 1
    # iterate over all the particles within the cells of index `idx`
    @inbounds for ip in cellaxes(Fp)
        # skip lines below if there is no particle in this piece of memory
        doskip(index, ip, idxCA...) && continue

        # cache particle coordinates
        pᵢ = ntuple(i -> (@index p[i][ip, idxCA...]), Val(N2))

        # skip lines below if there is no particle in this piece of memory
        # any(isnan, pᵢ) && continue

        ntuple(Val(N1)) do i
            Base.@_inline_meta
            Fᵢ = @index Fp[i][ip, idxCA...]
            F_pic, F0_pic = _grid2particle(pᵢ, xvi, di, (F[i], F0[i]), idx)
            ΔF = F_pic - F0_pic
            F_flip = Fᵢ + ΔF
            # Interpolate field F onto particle
            @index Fp[i][ip, idxCA...] = muladd(F_pic, α, F_flip * (1.0 - α))
        end
    end
    return nothing
end

#  Interpolation from grid corners to particle positions --------------------------------------------------------

@inline function _grid2particle(
        pᵢ::Union{SVector, NTuple}, xvi::NTuple, di::NTuple, F::AbstractArray, idx
    )
    # F at the cell corners
    Fi = field_corners(F, idx .- 2)
    # normalize particle coordinates
    ti = normalize_coordinates(pᵢ, xvi, di, idx)
    # Interpolate field F onto particle
    Fp = lerp(Fi, ti)

    return Fp
end

@inline function _grid2particle(
        pᵢ::Union{SVector, NTuple}, xvi::NTuple, di::NTuple, Fi::NTuple{N, Number}, idx
    ) where {N}
    # normalize particle coordinates
    ti = normalize_coordinates(pᵢ, xvi, di, idx)
    # Interpolate field F onto particle
    Fp = lerp(Fi, ti)

    return Fp
end

@inline function _grid2particle(
        pᵢ::Union{SVector, NTuple}, xvi::NTuple, di::NTuple, F::NTuple{N, AbstractArray}, idx
    ) where {N}
    # normalize particle coordinates
    ti = normalize_coordinates(pᵢ, xvi, di, idx)
    Fp = ntuple(Val(N)) do i
        Base.@_inline_meta
        # F at the cell corners
        Fi = field_corners(F[i], idx)
        # Interpolate field F onto particle
        lerp(Fi, ti)
    end

    return Fp
end

@inline function _grid2particle(
        pᵢ::Union{SVector, NTuple}, xvi::NTuple, di::NTuple, F::NTuple{N1, NTuple{N2, Number}}, idx
    ) where {N1, N2}
    # normalize particle coordinates
    ti = normalize_coordinates(pᵢ, xvi, di, idx)
    Fp = ntuple(Val(N1)) do i
        Base.@_inline_meta
        # Interpolate field F onto particle
        lerp(F[i], ti)
    end

    return Fp
end

@inline function _grid2particle(
        pᵢ::Union{SVector, NTuple}, xvi::NTuple{N1, T}, di::NTuple, Fi::NTuple{N2, T}
    ) where {N1, N2, T <: Real}
    # normalize particle coordinates
    ti = normalize_coordinates(pᵢ, xvi, di)
    # Interpolate field F onto particle
    Fp = lerp(Fi, ti)
    return Fp
end
