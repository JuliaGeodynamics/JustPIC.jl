# LAUNCHERS
function grid2particle!(Fp, xvi, F, particles::PassiveMarkers)
    (; coords, np) = particles
    dxi = grid_size(xvi)

    @parallel (@idx np) grid2particle_passive_marker!(Fp, F, xvi, dxi, coords)

    return nothing
end

@parallel_indices (ip) function grid2particle_passive_marker!(
        Fp, F, xvi, dxi, particle_coords
    )
    _grid2particle_passive_marker!(Fp, F, xvi, dxi, particle_coords, ip)
    return nothing
end

# INNERMOST INTERPOLATION KERNEL

@inline function _grid2particle_passive_marker!(
        Fp::AbstractArray, F::AbstractArray, xvi, dxi::NTuple{N}, p, ip
    ) where {N}

    # particle coordinates
    pᵢ = get_particle_coords(p, ip)
    # pᵢ = p[ip].data

    I = ntuple(Val(N)) do i
        Base.@_inline_meta
        cell_index(pᵢ[i], xvi[i], dxi[i])
    end

    Fi = field_corners(F, I)

    # Interpolate field F onto particle
    Fp[ip] = _grid2particle(pᵢ, xvi, dxi, Fi, I)

    return nothing
end

@inline function _grid2particle!(Fp, ip, pᵢ, xvi, dxi, Fi, I)
    # Interpolate field F onto particle
    return Fp[ip] = _grid2particle(pᵢ, xvi, dxi, Fi, I)
end

@inline function _grid2particle_passive_marker!(
        Fp::NTuple{N1, AbstractArray}, F::NTuple{N1, AbstractArray}, xvi, dxi::NTuple{N2}, p, ip
    ) where {N1, N2}

    # particle coordinates
    pᵢ = get_particle_coords(p, ip)

    I = ntuple(Val(N2)) do i
        Base.@_inline_meta
        cell_index(pᵢ[i], xvi[i], dxi[i])
    end

    ntuple(Val(N1)) do i
        Fi = field_corners(F[i], I)
        # Interpolate field F onto particle
        Fp[i][ip] = _grid2particle(pᵢ, xvi, dxi, Fi, I)
    end

    return nothing
end
