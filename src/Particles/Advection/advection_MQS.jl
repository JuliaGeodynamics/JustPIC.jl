"""
    advection_MQS!(particles, method, V, dt)

Advect particles using the monotonic quadratic spline (`MQS`) velocity
interpolation scheme.

Compared with `advection!`, this method reconstructs staggered velocities with
`MQS` where enough stencil support is available.

Near boundaries or when the required stencil is unavailable, the implementation
falls back to linear interpolation.

The public entry point reads the staggered velocity coordinates and spacing from
`particles.xi_vel` and `particles.di.velocity`.
"""
advection_MQS!(
        particles::Particles,
        method::AbstractAdvectionIntegrator,
        V,
        dt,
    ) = advection_MQS!(
        particles,
        method,
        V,
        particles.xi_vel,
        dt,
        particles.di.velocity
    ) 

function advection_MQS!(
        particles::Particles,
        method::AbstractAdvectionIntegrator,
        V,
        grid_vi::NTuple{N, NTuple{N}},
        dt,
        dxi
    ) where {N}
    interpolation_fn = interp_velocity2particle_MQS

    # dxi = compute_dx(first(grid_vi))
    (; coords, index) = particles
    # compute some basic stuff
    ni = size(index)
    # compute local limits (i.e. domain or MPI rank limits)
    local_limits = inner_limits(grid_vi)

    # launch parallel advection kernel
    @parallel (@idx ni) advection_kernel_MQS!(
        coords, method, V, index, grid_vi, local_limits, dxi, dt, interpolation_fn
    )

    return nothing
end

# DIMENSION AGNOSTIC KERNELS

@parallel_indices (I...) function advection_kernel_MQS!(
        p,
        method::AbstractAdvectionIntegrator,
        V::NTuple{N},
        index,
        grid,
        local_limits,
        dxi,
        dt,
        interpolation_fn::F,
    ) where {N, F}

    # iterate over particles in the I-th cell
    for ipart in cellaxes(index)
        # skip if particle does not exist in this memory location
        doskip(index, ipart, I...) && continue
        # extract particle coordinates
        pᵢ = get_particle_coords(p, ipart, I...)
        # # advect particle
        pᵢ_new = advect_particle(
            method, pᵢ, V, grid, local_limits, dxi, dt, interpolation_fn, I
        )
        # update particle coordinates
        for k in 1:N
            @inbounds @index p[k][ipart, I...] = pᵢ_new[k]
        end
    end

    return nothing
end

@inline function interp_velocity2particle_MQS(
        particle_coords::NTuple{N}, grid_vi, local_limits, dxi, V::NTuple{N}, idx::NTuple{N}
    ) where {N}
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        local_lims = local_limits[i]
        v = if check_local_limits(local_lims, particle_coords)
            interp_velocity2particle_MQS(particle_coords, grid_vi[i], dxi[i], V[i], Val(i), idx)
        else
            Inf
        end
    end
end

@inline function interp_velocity2particle_MQS(
        p_i::Union{SVector, NTuple}, xi_vx::NTuple, dxi::NTuple, F::AbstractArray, ::Val{N}, idx
    ) where {N}
    # F and coordinates of the cell corners
    Fi, xci, indices = corner_field_nodes_LinP(F, p_i, xi_vx, idx)
    # Recompute the local spacing from the corrected parent cell. On nonuniform
    # grids the seed cell `idx` may differ from the actual interpolation cell.
    di = @dxi(dxi, indices...)
    # normalize particle coordinates
    t = normalize_coordinates(p_i, xci, di)
    # Interpolate field F from pressure node onto particle
    V = if all(x -> 1 < x[1] < x[2] - 1, zip(indices, size(F)))
        MQS(F, Fi, t, indices..., Val(N))
    else
        lerp(Fi, t)
    end
    return V
end
