"""
    interpolate_velocity_to_markerchain!(chain::MarkerChain, chain_V::NTuple{N, CellArray}, V, grid_vi::NTuple{N, NTuple{N, T}}) where {N, T}

Interpolate the staggered velocity field `V` to the current marker positions in
`chain` and store the result in `chain_V`.

`chain_V` must be preallocated with the same cell layout as the marker-chain
coordinates.
"""
function interpolate_velocity_to_markerchain!(
        chain::MarkerChain,
        chain_V,
        V,
        grid_vi::NTuple{N, NTuple{N, T}},
    ) where {N, T}
    (; coords, index) = chain

    # recast the grid to the marker precision so the ranges are GPU-safe on Float32
    # backends (they are indexed directly inside the kernel; see advection!)
    Tc = eltype(eltype(coords[1]))
    grid_vi = recast_grid(grid_vi, Tc)

    # compute some basic stuff
    ni = size(index, 1)
    dxi = compute_dx(first(grid_vi))

    local_limits = inner_limits(grid_vi)

    # launch parallel advection kernel
    launch!(
        ka_backend(chain), interpolate_velocity_to_markerchain_kernel!, ni,
        coords, chain_V, V, index, grid_vi, local_limits, dxi,
    )
    return nothing
end

# Velocity interpolation kernel for marker chains on staggered grids.
@kernel function interpolate_velocity_to_markerchain_kernel!(
        p,
        chain_V,
        V::NTuple{N, T},
        index,
        grid,
        local_limits,
        dxi,
    ) where {N, T}
    i = @index(Global)

    for ipart in cellaxes(index)
        # skip if particle does not exist in this memory location
        doskip(index, ipart, i) && continue
        # extract particle coordinates
        pᵢ = get_particle_coords(p, ipart, i)
        # interpolate velocity to particle
        v = interp_velocity2particle_markerchain(pᵢ, grid, local_limits, dxi, V)
        CAI.@index chain_V[1][ipart, i] = v[1]
        CAI.@index chain_V[2][ipart, i] = v[2]
    end
end
