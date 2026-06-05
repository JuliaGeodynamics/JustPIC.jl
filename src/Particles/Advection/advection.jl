"""
    advection!(particles::Particles, method::AbstractAdvectionIntegrator, V, grid_vi::NTuple{N,NTuple{N,T}}, dt)

Advect `particles` through the staggered velocity field `V` over a time step `dt`.

`grid_vi` contains the coordinate tuples associated with each staggered velocity
component. The particle coordinates are updated in place.

# Arguments
- `particles`: `Particles` container to advect.
- `method`: time integrator such as `Euler()`, `RungeKutta2()`, or `RungeKutta4()`.
- `V`: tuple of staggered velocity component arrays.
- `grid_vi`: tuple of coordinate tuples matching the staggering of `V`.
- `dt`: timestep.
"""
advection!(
    particles::Particles,
    method::AbstractAdvectionIntegrator,
    V,
    dt,
) = advection!(
    particles,
    method,
    V,
    particles.xi_vel,
    dt,
    particles.di.velocity
)

function advection!(
        particles::Particles,
        method::AbstractAdvectionIntegrator,
        V,
        grid_vi::NTuple{N, NTuple{N, T}},
        dt,
        dxi
    ) where {N, T}
    (; coords, index) = particles
    # compute some basic stuff
    ni = size(index)
    # compute local limits (i.e. domain or MPI rank limits)
    local_limits = inner_limits(grid_vi)

    # launch parallel advection kernel
    @parallel (@idx ni) advection_kernel!(
        coords, method, V, index, grid_vi, local_limits, dxi, dt
    )

    return nothing
end

# DIMENSION AGNOSTIC KERNELS

@parallel_indices (I...) function advection_kernel!(
        p,
        method::AbstractAdvectionIntegrator,
        V::NTuple{N, T},
        index,
        grid_vi,
        local_limits,
        dxi,
        dt,
    ) where {N, T}

    # iterate over particles in the I-th cell
    for ipart in cellaxes(index)
        # skip if particle does not exist in this memory location
        doskip(index, ipart, I...) && continue
        # extract particle coordinates
        pᵢ = get_particle_coords(p, ipart, I...)
        # # advect particle
        pᵢ_new = advect_particle(method, pᵢ, V, grid_vi, local_limits, dxi, dt, I)
        # update particle coordinates
        for k in 1:N
            @index p[k][ipart, I...] = pᵢ_new[k]
        end
    end

    return nothing
end

@inline function interp_velocity2particle(
        particle_coords::NTuple{N, Any},
        grid_vi,
        local_limits,
        dxi,
        V::NTuple{N, Any},
        idx::NTuple{N, Any},
    ) where {N}
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        local_lims = local_limits[i]
        v = if check_local_limits(local_lims, particle_coords)
            interp_velocity2particle(particle_coords, grid_vi[i], dxi[i], V[i], idx)
        else
            Inf
        end
    end
end

# Interpolate velocity from staggered grid to particle. Innermost kernel
@inline function interp_velocity2particle(
        p_i::Union{SVector, NTuple}, xi_vx::NTuple, di::NTuple, F::AbstractArray, idx
    )
    # F and coordinates at/of the cell corners
    Fi, xci, indices = corner_field_nodes(F, p_i, xi_vx, idx)
    # normalize particle coordinates
    dxi = @dxi di indices...
    ti = normalize_coordinates(p_i, xci, dxi)
    # Interpolate field F onto particle
    Fp = lerp(Fi, ti)
    # return interpolated field
    return Fp
end

@generated function corner_field_nodes(
        F::AbstractArray{T, N},
        particle,
        xi_vx,
        idx
    ) where {T, N}
    return quote
        @inline
        Base.@nexprs $N i -> begin
            corrected_idx_i = find_parent_cell_bisection(particle[i], xi_vx[i], idx[i])
            cell_i = xi_vx[i][corrected_idx_i]
        end

        indices = Base.@ncall $N tuple corrected_idx
        xci = Base.@ncall $N tuple cell

        # F at the four centers
        Fi = extract_field_corners(F, indices...)

        return Fi, xci, indices
    end
end

@inline function vertex_offset(xi, pxi, di)
    dist = normalised_distance(xi, pxi, di)
    return (dist > 2) * 2 + (2 > dist > 1) * 1 + (-1 < dist < 0) * -1 + (dist < -1) * -2
end
