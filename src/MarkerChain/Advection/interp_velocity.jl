"""
    interpolate_velocity_to_markerchain!(chain::MarkerChain, chain_V::NTuple{N, CellArray}, V, grid_vi::NTuple{N, NTuple{N, T}}) where {N, T}

Interpolates the velocity field to the positions of the marker chain.

# Arguments
- `chain::MarkerChain`: The marker chain object containing the particle coordinates and indices.
- `chain_V::NTuple{N, CellArray}`: The output velocity field at the marker chain positions.
- `V`: The velocity field to be interpolated.
- `grid_vi::NTuple{N, NTuple{N, T}}`: The grid information for each dimension.
"""
function interpolate_velocity_to_markerchain!(
        chain::MarkerChain,
        chain_V,
        V,
        grid_vi::NTuple{N, NTuple{N, T}},
    ) where {N, T}
    (; coords, index) = chain

    # compute some basic stuff
    ni = size(index, 1)
    dxi = compute_dx(first(grid_vi))

    local_limits = inner_limits(grid_vi)

    # launch parallel advection kernel
    @parallel (@idx ni) interpolate_velocity_to_markerchain_kernel!(
        coords, chain_V, V, index, grid_vi, local_limits, dxi,
    )
    return nothing
end

# ParallelStencil function Runge-Kuttaadvection function for 3D staggered grids
@parallel_indices (i) function interpolate_velocity_to_markerchain_kernel!(
        p,
        chain_V,
        V::NTuple{N, T},
        index,
        grid,
        local_limits,
        dxi,
    ) where {N, T}
    for ipart in cellaxes(index)
        # skip if particle does not exist in this memory location
        doskip(index, ipart, i) && continue
        # extract particle coordinates
        pᵢ = get_particle_coords(p, ipart, i)
        # interpolate velocity to particle
        v = interp_velocity2particle_markerchain(pᵢ, grid, local_limits, dxi, V)
        @index chain_V[1][ipart, i] = v[1]
        @index chain_V[2][ipart, i] = v[2]
    end

    return nothing
end
