# Main Runge-Kutta advection function for 2D staggered grids
"""
    advection!(particles::Particles, method::AbstractAdvectionIntegrator, V, grid_vi::NTuple{N,NTuple{N,T}}, dt)

Advects the particles using the advection scheme defined by `method`.

# Arguments
- `particles`: Particles object to be advected.
- `method`: Time integration method (`Euler` or `RungeKutta2`).
- `V`: Tuple containing `Vx`, `Vy`; and `Vz` in 3D.
- `grid_vi`: Tuple containing the grids corresponding to `Vx`, `Vy`; and `Vz` in 3D.
- `dt`: Time step.
"""
function advection!(
        particles::Particles,
        method::AbstractAdvectionIntegrator,
        V,
        grid_vi::NTuple{N, NTuple{N, T}},
        dt,
    ) where {N, T}
    dxi = compute_dx(first(grid_vi))
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
        grid,
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
        pᵢ_new = advect_particle(method, pᵢ, V, grid, local_limits, dxi, dt, I)
        # update particle coordinates
        for k in 1:N
            @inbounds @index p[k][ipart, I...] = pᵢ_new[k]
        end
    end

    return nothing
end

@inline function interp_velocity2particle(
        particle_coords::NTuple{N, Any},
        grid,
        local_limits,
        dxi,
        V::NTuple{N, Any},
        idx::NTuple{N, Any},
    ) where {N}
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        local_lims = local_limits[i]
        v = if check_local_limits(local_lims, particle_coords)
            interp_velocity2particle(particle_coords, grid_vi[i], dxi, V[i], idx)
        else
            Inf
        end
    end
end

@generated function corner_field_nodes(
        F::AbstractArray{T, N},
        particle,
        xi_vx,
        dxi,
        idx::Union{SVector{N, Integer}, NTuple{N, Integer}},
    ) where {N, T}
    return quote
        Base.@_inline_meta
        begin
            Base.@nexprs $N i -> begin
                corrected_idx_i = cell_index(particle[i], xi_vx[i])
                cell_i = @inbounds xi_vx[i][corrected_idx_i]
            end

            indices = Base.@ncall $N tuple corrected_idx
            cells = Base.@ncall $N tuple cell

            # F at the four centers
            Fi = extract_field_corners(F, indices...)
        end

        return Fi, cells
    end
end

@inline function vertex_offset(xi, pxi, di)
    dist = normalised_distance(xi, pxi, di)
    return (dist > 2) * 2 + (2 > dist > 1) * 1 + (-1 < dist < 0) * -1 + (dist < -1) * -2
end
