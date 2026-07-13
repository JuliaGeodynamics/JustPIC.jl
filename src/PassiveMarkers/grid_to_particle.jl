# LAUNCHERS
"""
    grid2particle!(Fp, xvi, F, particles::PassiveMarkers)

Interpolate a nodal field `F` to passive-marker values `Fp`, updated in place.

The vertex grid `xvi` must be supplied explicitly, since `PassiveMarkers` stores
only marker coordinates and no grid metadata.

# Arguments
- `Fp`: destination marker field, or tuple of marker fields.
- `xvi`: vertex coordinates of the grid on which `F` is defined.
- `F`: source nodal field, or tuple of nodal fields matching `Fp`.
- `particles`: `PassiveMarkers` container supplying marker coordinates.
"""
function grid2particle!(Fp, xvi, F, particles::PassiveMarkers)
    (; coords, np) = particles
    # recast the grid to the marker precision so the ranges are GPU-safe on Float32
    # backends (they are indexed directly inside the kernel; see advection!)
    xvi = recast_grid(xvi, eltype(coords[1]))
    dxi = grid_size(xvi)

    launch!(ka_backend(particles), grid2particle_passive_marker!, np, Fp, F, xvi, dxi, coords)

    return nothing
end

@kernel function grid2particle_passive_marker!(
        Fp, F, xvi, dxi, particle_coords
    )
    ip = @index(Global)
    _grid2particle_passive_marker!(Fp, F, xvi, dxi, particle_coords, ip)
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
