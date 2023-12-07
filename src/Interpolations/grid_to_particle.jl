## CLASSIC PIC ------------------------------------------------------------------------------------------------

# LAUNCHERS
function grid2particle!(Fp, xvi, F, particle_coords, index)
    di = grid_size(xvi)
    grid2particle!(Fp, xvi, F, particle_coords, index, di)
    return nothing
end

function grid2particle!(Fp, xvi, F, particle_coords, index, di::NTuple{N,T}) where {N,T}
    ni = length.(xvi)

    @parallel (@idx ni .- 1) grid2particle_classic!(Fp, F, xvi, index, di, particle_coords)

    return nothing
end

@parallel_indices (I...) function grid2particle_classic!(Fp, F, xvi, index, di, particle_coords)
    _grid2particle_classic!(Fp, particle_coords, xvi, di, F, index, tuple(I...))
    return nothing
end

# INNERMOST INTERPOLATION KERNEL

@inline function _grid2particle_classic!(Fp, p, xvi, di::NTuple{N,T}, F, index, idx) where {N,T}
    # iterate over all the particles within the cells of index `idx` 
    @inbounds for ip in cellaxes(Fp)
        # skip lines below if there is no particle in this pice of memory
        !(@cell index[ip, idx...]) && continue

        # cache particle coordinates 
        # pᵢ = ntuple(i -> (@cell p[i][ip, idx...]), Val(N))
        pᵢ = get_particle_coords(p, ip, idx...)

        # Interpolate field F onto particle
        new_value = _grid2particle(pᵢ, xvi, di, F, idx)
        # @inline @cell Fp[ip, idx...] = new_value
        setelement!(Fp, new_value, ip, idx...)
    end
end

@inline function _grid2particle_classic!(
    Fp::NTuple{N1,T1}, p, xvi, di::NTuple{N2,T2}, F::NTuple{N1,T3}, index, idx
) where {N1,T1,N2,T2,T3}
    # iterate over all the particles within the cells of index `idx` 
    @inbounds for ip in cellaxes(Fp[1])
        # skip lines below if there is no particle in this pice of memory
        !(@cell index[ip, idx...]) && continue

        # cache particle coordinates 
        # pᵢ = ntuple(i -> (@cell p[i][ip, idx...]), Val(N))
        pᵢ = get_particle_coords(p, ip, idx...)

        # skip lines below if there is no particle in this pice of memory
        any(isnan, pᵢ) && continue

        # Interpolate field F onto particle
        ntuple(Val(N1)) do i
            Base.@_inline_meta
            @cell Fp[i][ip, idx...] = _grid2particle(pᵢ, xvi, di, F[i], idx)
        end
    end
end

#  Interpolation from grid corners to particle positions

@inline function _grid2particle(pᵢ::Union{SVector, NTuple}, xvi::NTuple, di::NTuple, F::AbstractArray, idx)
    # F at the cell corners
    Fi = field_corners(F, idx)
    # normalize particle coordinates
    ti = normalize_coordinates(pᵢ, xvi, di, idx)
    # Interpolate field F onto particle
    Fp = ndlinear(ti, Fi)

    return Fp
end

@inline function _grid2particle(
    pᵢ::Union{SVector, NTuple}, xvi::NTuple, di::NTuple, F::NTuple{N,T}, idx
) where {N,T}
    # normalize particle coordinates
    ti = normalize_coordinates(pᵢ, xvi, di, idx)
    Fp = ntuple(Val(N)) do i
        Base.@_inline_meta
        # F at the cell corners
        Fi = field_corners(F[i], idx)
        # Interpolate field F onto particle
        ndlinear(ti, Fi)
    end

    return Fp
end

## FULL PARTICLE PIC ------------------------------------------------------------------------------------------

# LAUNCHERS

function grid2particle_flip!(Fp, xvi, F, F0, particle_coords, index; α=0.0)
    di = grid_size(xvi)
    grid2particle_flip!(Fp, xvi, F, F0, particle_coords, index, di; α=α)

    return nothing
end

function grid2particle_flip!(
    Fp, xvi, F, F0, particle_coords, index, di::NTuple{N,T}; α=0.0
) where {N,T}
    ni = length.(xvi)

    @parallel (@idx ni .- 1) grid2particle_full!(Fp, F, F0, xvi, di, particle_coords, index, α)

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
    Fp, p, xvi, di::NTuple{N,T}, F, F0, index, idx, α
) where {N,T}
    # iterate over all the particles within the cells of index `idx` 
    @inbounds for ip in cellaxes(Fp)
        # skip lines below if there is no particle in this pice of memory
        !(@cell index[ip, idx...]) && continue

        # cache particle coordinates 
        # pᵢ = ntuple(i -> (@cell p[i][ip, idx...]), Val(N))
        pᵢ = get_particle_coords(p, ip, idx...)
        
        # # skip lines below if there is no particle in this pice of memory
        # any(isnan, pᵢ) && continue

        Fᵢ = @cell Fp[ip, idx...]
        F_pic, F0_pic = _grid2particle(pᵢ, xvi, di, (F, F0), idx)
        ΔF = F_pic - F0_pic
        F_flip = Fᵢ + ΔF
        # Interpolate field F onto particle
        @cell Fp[ip, idx...] = muladd(F_pic, α, F_flip * (1.0 - α))
    end
end

@inline function _grid2particle_full!(
    Fp::NTuple{N1,T1},
    p,
    xvi,
    di::NTuple{N2,T2},
    F::NTuple{N1,T3},
    F0::NTuple{N1,T3},
    index,
    idx,
    α,
) where {N1,T1,N2,T2,T3}
    # iterate over all the particles within the cells of index `idx` 
    @inbounds for ip in cellaxes(Fp)
        # skip lines below if there is no particle in this pice of memory
        !(@cell index[ip, idx...]) && continue

        # cache particle coordinates 
        pᵢ = ntuple(i -> (@cell p[i][ip, idx...]), Val(N2))

        # skip lines below if there is no particle in this pice of memory
        # any(isnan, pᵢ) && continue

        ntuple(Val(N1)) do i
            Base.@_inline_meta
            Fᵢ = @cell Fp[i][ip, idx...]
            F_pic, F0_pic = _grid2particle(pᵢ, xvi, di, (F[i], F0[i]), idx)
            ΔF = F_pic - F0_pic
            F_flip = Fᵢ + ΔF
            # Interpolate field F onto particle
            @cell Fp[i][ip, idx...] = muladd(F_pic, α, F_flip * (1.0 - α))
        end
    end
end


