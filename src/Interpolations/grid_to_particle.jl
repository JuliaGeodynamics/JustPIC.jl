## CLASSIC PIC ------------------------------------------------------------------------------------------------

# LAUNCHERS

function grid2particle!(Fp::AbstractArray, xvi, F::AbstractArray, particle_coords)
    di = grid_size(xvi)
    ni = length.(xvi)

    @parallel (@idx ni .- 1) grid2particle_classic!(Fp, F, xvi, di, particle_coords)

    return nothing
end

function grid2particle!(Fp::AbstractArray, xvi, F::AbstractArray, particle_coords, di)
    ni = length.(xvi)

    @parallel (@idx ni .- 1) grid2particle_classic!(Fp, F, xvi, di, particle_coords)

    return nothing
end

@parallel_indices (inode, jnode) function grid2particle_classic!(
    Fp, F, xvi, di::NTuple{2,Any}, particle_coords
)
    _grid2particle_classic!(Fp, particle_coords, xvi, di, F, (inode, jnode))
    return nothing
end

@parallel_indices (inode, jnode, knode) function grid2particle_classic!(
    Fp, F, xvi, di::NTuple{3,Any}, particle_coords
)
    _grid2particle_classic!(Fp, particle_coords, xvi, di, F, (inode, jnode, knode))
    return nothing
end

# INNERMOST INTERPOLATION KERNEL

@inline function _grid2particle_classic!(Fp, p, xvi, di::NTuple{N,T}, F, idx) where {N,T}
    # iterate over all the particles within the cells of index `idx` 
    for ip in cellaxes(Fp)
        # cache particle coordinates 
        pᵢ = ntuple(i -> (@cell p[i][ip, idx...]), Val(N))

        # skip lines below if there is no particle in this pice of memory
        any(isnan, pᵢ) && continue

        # Interpolate field F onto particle
        @cell Fp[ip, idx...] = _grid2particle(pᵢ, xvi, di, F, idx)
    end
end

#  Interpolation from grid corners to particle positions

function _grid2particle(pᵢ::NTuple, xvi::NTuple, di::NTuple, F::AbstractArray, idx)
    # F at the cell corners
    Fi = field_corners(F, idx)
    # normalize particle coordinates
    ti = normalize_coordinates(pᵢ, xvi, di, idx)
    # Interpolate field F onto particle
    Fp = ndlinear(ti, Fi)

    return Fp
end

## FULL PARTICLE PIC ------------------------------------------------------------------------------------------

# LAUNCHERS

function grid2particle!(
    Fp::AbstractArray, xvi, F::AbstractArray, F0::AbstractArray, particle_coords; α=0.0
)
    di = grid_size(xvi)
    ni = length.(xvi)

    @parallel (@idx ni .- 1) grid2particle_full!(Fp, F, F0, xvi, di, particle_coords, α)

    return nothing
end

function grid2particle!(
    Fp::AbstractArray, xvi, F::AbstractArray, F0::AbstractArray, particle_coords, di; α=0.0
)
    ni = length.(xvi)

    @parallel (@idx ni .- 1) grid2particle_full!(Fp, F, F0, xvi, di, particle_coords, α)

    return nothing
end

@parallel_indices (inode, jnode) function grid2particle_full!(
    Fp, F, F0, xvi, di::NTuple{2,Any}, particle_coords, α
)
    _grid2particle_full!(Fp, particle_coords, xvi, di, F, F0, (inode, jnode), α)
    return nothing
end

@parallel_indices (inode, jnode, knode) function grid2particle_full!(
    Fp, F, F0, xvi, di::NTuple{3,Any}, particle_coords, α
)
    _grid2particle_full!(Fp, particle_coords, xvi, di, F, F0, (inode, jnode, knode), α)
    return nothing
end

# INNERMOST INTERPOLATION KERNEL

@inline function _grid2particle_full!(
    Fp, p, xvi, di::NTuple{N,T}, F, F0, idx, α
) where {N,T}
    # iterate over all the particles within the cells of index `idx` 
    for ip in cellaxes(Fp)
        # cache particle coordinates 
        pᵢ = ntuple(i -> (@cell p[i][ip, idx...]), Val(N))

        # skip lines below if there is no particle in this pice of memory
        any(isnan, pᵢ) && continue

        Fᵢ = @cell Fp[ip, idx...]
        F_pic = _grid2particle(pᵢ, xvi, di, F, idx)
        ΔF = F_pic - _grid2particle(pᵢ, xvi, di, F0, idx)
        F_flip = Fᵢ + ΔF
        # Interpolate field F onto particle
        @cell Fp[ip, idx...] = F_pic * α + F_flip * (1.0 - α)
    end
end
