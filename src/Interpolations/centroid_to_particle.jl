## CLASSIC PIC ------------------------------------------------------------------------------------------------

# LAUNCHERS
function centroid2particle!(Fp, xci, F, particles)
    (; coords) = particles
    di = grid_size(xci)
    centroid2particle!(Fp, xci, F, coords, di)
    return nothing
end

function centroid2particle!(Fp, xci, F, coords, di::NTuple{N,T}) where {N,T}
    indices = ntuple(i -> 0:(length(xci[i]) + 1), Val(N))

    @parallel (indices) centroid2particle_classic!(Fp, F, xci, di, coords)

    return nothing
end

@parallel_indices (I...) function centroid2particle_classic!(Fp, F, xci, di, coords)
    _centroid2particle_classic!(Fp, coords, xci, di, F, tuple(I...))
    return nothing
end

# INNERMOST INTERPOLATION KERNEL

@inline function _centroid2particle_classic!(Fp, p, xci, di::NTuple{2,T}, F, idx) where {T}
    nx, ny = size(F)
    idx_i, idx_j = idx

    for j in idx_j:(idx_j + 1)
        !(1 ≤ j ≤ ny) && continue
        for i in idx_i:(idx_i + 1)
            !(1 ≤ i ≤ nx) && continue

            # iterate over all the particles within the cells of index `idx`
            for ip in cellaxes(Fp)
                # cache particle coordinates
                pᵢ = ntuple(ii -> (@cell p[ii][ip, i, j]), Val(2))
                xc = xci[1][i], xci[2][j]
                cell_index = shifted_index(pᵢ, xc, idx)
                # skip lines below if there is no particle in this piece of memory
                any(isnan, pᵢ) && continue
                # Interpolate field F onto particle
                @cell Fp[ip, i, j] = _centroid2particle(pᵢ, xci, di, F, cell_index)
            end
        end
    end
    return nothing
end

@inline function _centroid2particle_classic!(
    Fp::NTuple{N,T}, p, xci, di::NTuple{2,T}, F, idx
) where {N,T}
    nx, ny = size(F)
    idx_i, idx_j = idx

    for j in idx_j:(idx_j + 1)
        !(1 ≤ j ≤ ny) && continue
        for i in idx_i:(idx_i + 1)
            !(1 ≤ i ≤ nx) && continue

            # iterate over all the particles within the cells of index `idx`
            for ip in cellaxes(Fp)
                # cache particle coordinates
                pᵢ = ntuple(ii -> (@cell p[ii][ip, i, j]), Val(2))
                xc = xci[1][i], xci[2][j]
                cell_index = shifted_index(pᵢ, xc, idx)
                # skip lines below if there is no particle in this piece of memory
                any(isnan, pᵢ) && continue
                # Interpolate field F onto particle
                ntuple(Val(N)) do n
                    Base.@_inline_meta
                    @cell Fp[n][ip, i, j] = _centroid2particle(
                        pᵢ, xci, di, F[n], cell_index
                    )
                end
            end
        end
    end
    return nothing
end

@inline function _centroid2particle_classic!(Fp, p, xci, di::NTuple{3,T}, F, idx) where {T}
    nx, ny, nz = size(F)
    idx_i, idx_j, idx_k = idx

    for k in idx_k:(idx_k + 1)
        !(1 ≤ k ≤ nz) && continue
        for j in idx_j:(idx_j + 1)
            !(1 ≤ j ≤ ny) && continue
            for i in idx_i:(idx_i + 1)
                !(1 ≤ i ≤ nx) && continue

                # iterate over all the particles within the cells of index `idx`
                for ip in cellaxes(Fp)
                    # cache particle coordinates
                    pᵢ = ntuple(ii -> (@cell p[ii][ip, i, j, k]), Val(3))
                    xc = xci[1][i], xci[2][j], xci[3][k]
                    cell_index = shifted_index(pᵢ, xc, idx)
                    # skip lines below if there is no particle in this piece of memory
                    any(isnan, pᵢ) && continue
                    # Interpolate field F onto particle
                    @cell Fp[ip, i, j, k] = _centroid2particle(pᵢ, xci, di, F, cell_index)
                end
            end
        end
    end
    return nothing
end

@inline function _centroid2particle_classic!(
    Fp::NTuple{N,T}, p, xci, di::NTuple{3,T}, F, idx
) where {N,T}
    nx, ny, nz = size(F)
    idx_i, idx_j, idx_k = idx

    for k in idx_k:(idx_k + 1)
        !(1 ≤ k ≤ nz) && continue
        for j in idx_j:(idx_j + 1)
            !(1 ≤ j ≤ ny) && continue
            for i in idx_i:(idx_i + 1)
                !(1 ≤ i ≤ nx) && continue

                # iterate over all the particles within the cells of index `idx`
                for ip in cellaxes(Fp)
                    # cache particle coordinates
                    pᵢ = ntuple(ii -> (@cell p[ii][ip, i, j, k]), Val(3))
                    xc = xci[1][i], xci[2][j], xci[3][k]
                    cell_index = shifted_index(pᵢ, xc, idx)
                    # skip lines below if there is no particle in this piece of memory
                    any(isnan, pᵢ) && continue
                    # Interpolate field F onto particle
                    ntuple(Val(N)) do n
                        Base.@_inline_meta
                        @cell Fp[n][ip, i, j, k] = _centroid2particle(
                            pᵢ, xci, di, F[n], cell_index
                        )
                    end
                end
            end
        end
    end
    return nothing
end

## UTILS ------------------------------------------------------------------------------------------------------

#  Interpolation from grid corners to particle positions
@inline function _centroid2particle(
    pᵢ::NTuple, xci::NTuple, di::NTuple, F::AbstractArray, idx
)

    # F at the cell corners
    Fi = field_corners_clamped(F, idx)
    # normalize particle coordinates
    ti = normalize_coordinates_clamped(pᵢ, xci, di, idx)
    # Interpolate field F onto particle
    Fp = lerp(Fi, ti)

    return Fp
end

@inline function _centroid2particle(
    pᵢ::NTuple, xci::NTuple, di::NTuple, F::NTuple{N,T}, idx
) where {N,T}

    # normalize particle coordinates
    ti = normalize_coordinates_clamped(pᵢ, xci, di, idx)
    Fp = ntuple(Val(N)) do i
        Base.@_inline_meta
        # F at the cell corners
        Fi = field_corners_clamped(F[i], idx)
        # Interpolate field F onto particle
        lerp(Fi, ti)
    end

    return Fp
end

# Get field F at the corners of a given cell
@inline function field_corners_clamped(
    F::AbstractArray{T,2}, idx::NTuple{2,Integer}
) where {T}
    ni = nx, ny = size(F)
    idx2 = @inline ntuple(i -> clamp(idx[i], 1, ni[i]), Val(2))
    i, j = idx2
    i1, j1 = clamp(i + 1, 1, nx), clamp(j + 1, 1, ny)
    # optimal order for memory access
    F00 = F[i, j]
    F10 = F[i1, j]
    F01 = F[i, j1]
    F11 = F[i1, j1]
    # reorder to match the order of the lerp kernel
    return F00, F10, F01, F11
end

@inline function field_corners_clamped(
    F::AbstractArray{T,3}, idx::NTuple{3,Integer}
) where {T}
    ni = nx, ny, nz = size(F)
    idx2 = @inline ntuple(i -> clamp(idx[i], 1, ni[i]), Val(3))
    i, j, k = idx2
    i1, j1, k1 = clamp(i + 1, 1, nx), clamp(j + 1, 1, ny), clamp(k + 1, 1, nz)
    # optimal order for memory access
    F000 = F[i, j, k]    # v000
    F100 = F[i1, j, k]   # v100
    F010 = F[i, j1, k]   # v010
    F110 = F[i1, j1, k]  # v110
    F001 = F[i, j, k1]   # v001
    F101 = F[i1, j, k1]  # v101
    F011 = F[i, j1, k1]  # v011
    F111 = F[i1, j1, k1] # v111
    # reorder to match the order of the lerp kernel
    return F000, F100, F010, F110, F001, F101, F011, F111
end

# normalize coordinates
@inline function normalize_coordinates_clamped(
    p::NTuple{N,A}, xi::NTuple{N,B}, di::NTuple{N,C}, idx::NTuple{N,D}
) where {N,A,B,C,D}
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        j = idx[i]
        xc = if j < 1 # set cellcenter coordinates outside the domain otherwise
            -di[i] * 0.5
        else
            xi[i][j]
        end
        (p[i] - xc) * inv(di[i])
    end
end

# normalize coordinates
@inline function normalize_coordinates_clamped(
    p::NTuple{N,A}, xci::NTuple{N,B}, di::NTuple{N,C}
) where {N,A,B,C}
    return ntuple(i -> (p[i] - xci[i]) * inv(di[i]), Val(N))
end

# shifts the index of the cell bot-left to the left if it is located in the left cell
@inline shifted_index(pxi, xci, idx) = pxi < xci ? idx - 1 : idx
@inline shifted_index(pxi::NTuple{N,A}, xci::NTuple{N,B}, idx::NTuple{N,Integer}) where {N,A,B} = ntuple(i -> shifted_index(pxi[i], xci[i], idx[i]), Val(N))
