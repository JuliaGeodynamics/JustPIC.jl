## CLASSIC PIC ------------------------------------------------------------------------------------------------

# LAUNCHERS
function centroid2particle!(Fp, xci, F, particles)
    (; coords) = particles
    di = grid_size(xci)
    centroid2particle!(Fp, xci, F, coords, di)
    return nothing
end

function centroid2particle!(Fp, xci, F, coords, di::NTuple{N,T}) where {N,T}
    # indices = ntuple(i -> 0:(length(xci[i]) + 1), Val(N))
    ni = size(Fp)
    @parallel (@idx ni) centroid2particle_classic!(Fp, F, xci, di, coords)

    return nothing
end

@parallel_indices (I...) function centroid2particle_classic!(Fp, F, xci, di, coords)
    _centroid2particle_classic!(Fp, coords, xci, di, F, tuple(I...))
    return nothing
end

# INNERMOST INTERPOLATION KERNEL

@inline function _centroid2particle_classic!(Fp, p, xci, di::NTuple{N}, F, I) where {N}
    ni = size(F) .- 1
    # iterate over all the particles within the cells of index `idx`
    @inbounds for ip in cellaxes(Fp)
        # cache particle coordinates
        pᵢ = ntuple(i -> (@index p[i][ip, I...]), Val(N))
        # skip lines below if there is no particle in this piece of memory
        any(isnan, pᵢ) && continue
        # continue the kernel
        xc = ntuple(i -> xci[i][I[i]], Val(N))
        cell_index = shifted_index(pᵢ, xc, I)
        cell_index = clamp.(cell_index, 1, ni)
        # Interpolate field F onto particle
        @index Fp[ip, I...] = _grid2particle(pᵢ, xci, di, F, cell_index)
    end
    return nothing
end

@inline function _centroid2particle_classic!(
    Fp::NTuple{NF}, p, xci, di::NTuple{N}, F::NTuple{NF}, I
) where {NF,N}
    ni = size(F) .- 1
    # iterate over all the particles within the cells of index `idx`
    @inbounds for ip in cellaxes(Fp)
        # cache particle coordinates
        pᵢ = ntuple(i -> (@index p[i][ip, I...]), Val(N))
        # skip lines below if there is no particle in this piece of memory
        any(isnan, pᵢ) && continue
        # continue the kernel
        xc = ntuple(i -> xci[i][I[i]], Val(N))
        cell_index = shifted_index(pᵢ, xc, I)
        cell_index = lamp.(cell_index, 1, ni)
        # Interpolate field F onto particle
        for n in 1:NF # should be unrolled
            @index Fp[n][ip, I...] = _grid2particle(pᵢ, xci, di, F[n], cell_index)
        end
    end
    return nothing
end

## UTILS ------------------------------------------------------------------------------------------------------

# shifts the index of the cell bot-left to the left if it is located in the left cell
@inline shifted_index(pxi, xci, idx) = pxi < xci ? idx - 1 : idx
@inline shifted_index(
    pxi::NTuple{N,A}, xci::NTuple{N,B}, idx::NTuple{N,Integer}
) where {N,A,B} = ntuple(i -> shifted_index(pxi[i], xci[i], idx[i]), Val(N))
