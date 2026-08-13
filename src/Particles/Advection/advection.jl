"""
    advection!(particles::Particles, method::AbstractAdvectionIntegrator, V, dt)
    advection!(particles::Particles, method::AbstractAdvectionIntegrator, V, grid_vi, dt, dxi)

Advect `particles` through the staggered velocity field `V` over a time step `dt`.
The particle coordinates are updated in place.

The public form reads the staggered velocity coordinate grids and spacing from
`particles` (`particles.xi_vel` and `particles.di.velocity`), so only `V` and
`dt` are supplied. The lower-level form takes those grids explicitly.

# Arguments
- `particles`: `Particles` container to advect.
- `method`: time integrator such as `Euler()`, `RungeKutta2()`, or `RungeKutta4()`.
- `V`: tuple of staggered velocity component arrays.
- `dt`: timestep.
- `grid_vi`: tuple of coordinate tuples matching the staggering of `V`
  (lower-level form only).
- `dxi`: grid spacing associated with `grid_vi` (lower-level form only).
- `periodic_1`, `periodic_2`, `periodic_3`: enable periodic wrapping at every
  integration stage in the corresponding coordinate direction.

# Notes
- Use the same periodic keywords in the subsequent `move_particles!` call.
- Stage-wise wrapping is required by `RungeKutta2` and `RungeKutta4`, whose
  intermediate interpolation points may cross a periodic boundary.
"""
advection!(
    particles::Particles,
    method::AbstractAdvectionIntegrator,
    V,
    dt,
    ;
    periodic_1 = false,
    periodic_2 = false,
    periodic_3 = false,
) = advection!(
    particles,
    method,
    V,
    particles.xi_vel,
    dt,
    particles.di.velocity;
    periodic_1 = periodic_1,
    periodic_2 = periodic_2,
    periodic_3 = periodic_3,
)

function advection!(
        particles::Particles,
        method::AbstractAdvectionIntegrator,
        V,
        grid_vi::NTuple{N, NTuple{N, T}},
        dt,
        dxi;
        periodic_1 = false,
        periodic_2 = false,
        periodic_3 = false,
    ) where {N, T}
    (; coords, index) = particles
    N == 2 && periodic_3 && throw(ArgumentError("periodic_3 is only valid for 3D particles"))
    # compute some basic stuff
    ni = inner_size(index)
    # compute local limits (i.e. domain or MPI rank limits)
    local_limits = inner_limits(grid_vi)
    periodicity = periodic_1, periodic_2, periodic_3
    domain_limits = physical_domain_limits(particles)

    # recast the integrator/timestep to the particle precision so Float32 backends
    # (e.g. Metal) don't carry a Float64 value into the kernel
    Tc = eltype(eltype(coords[1]))
    method = set_precision(method, Tc)
    dt = convert(Tc, dt)

    # launch parallel advection kernel
    launch!(
        ka_backend(particles), advection_kernel!, ni,
        coords, method, V, index, grid_vi, local_limits, dxi, dt, periodicity, domain_limits
    )

    return nothing
end

# DIMENSION AGNOSTIC KERNELS

@kernel function advection_kernel!(
        p,
        method::AbstractAdvectionIntegrator,
        V::NTuple{N, T},
        index,
        grid_vi,
        local_limits,
        dxi,
        dt,
        periodicity,
        domain_limits,
    ) where {N, T}
    I = @index(Global, NTuple)
    I_inner = I .+ 1

    # iterate over particles in the I-th cell
    for ipart in cellaxes(index)
        # skip if particle does not exist in this memory location
        doskip(index, ipart, I_inner...) && continue
        # extract particle coordinates
        pᵢ = get_particle_coords(p, ipart, I_inner...)
        # advect particle
        pᵢ_new = advect_particle(
            method, pᵢ, V, grid_vi, local_limits, dxi, dt, I_inner, periodicity, domain_limits
        )
        # update particle coordinates
        for k in 1:N
            CAI.@index p[k][ipart, I_inner...] = pᵢ_new[k]
        end
    end
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
            convert(eltype(V[i]), Inf)
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
