# LAUNCHERS

function grid2particle!(Fp::AbstractArray, xvi, F, particle_coords)
    di = grid_size(xvi)
    ni = length.(xvi)

    @parallel (@idx ni.-1) grid2particle!(Fp, F, xvi, di, particle_coords)

end

@parallel_indices (inode, jnode) function grid2particle!(Fp, F, xvi, di::NTuple{2, Any}, particle_coords)
    inner_grid2particle!(
        Fp, particle_coords, xvi, di, F, (inode, jnode)
    )
    return nothing
end

@parallel_indices (inode, jnode, knode) function grid2particle!(Fp, F, xvi, di::NTuple{3, Any}, particle_coords)
    inner_grid2particle!(
        Fp, particle_coords, xvi, di, F, (inode, jnode, knode)
    )
    return nothing
end

# INNERMOST INTERPOLATION KERNEL

@inline function inner_grid2particle!(Fp, p, xvi, di::NTuple{N, T}, F, idx) where {N, T}
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

function _grid2particle(
    pᵢ::NTuple, xvi::NTuple, di::NTuple, F::AbstractArray, idx
)
    # F at the cell corners
    Fi = field_corners(F, idx)
    # normalize particle coordinates
    ti = normalize_coordinates(pᵢ, xvi, di, idx)
    # Interpolate field F onto particle
    Fp = ndlinear(ti, Fi)
    
    return Fp
end