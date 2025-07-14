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
    v = interp_velocity2particle(particle_coords, grid, dxi, V, idx)
    return ntuple(Val(N)) do i
        Base.@_inline_meta
        local_lims = local_limits[i]
        check_local_limits(local_lims, particle_coords) ? v[i] : Inf
    end
end

@inline function interp_velocity2particle(
        p_i::Union{SVector, NTuple}, grid::NTuple{2}, dxi::NTuple{2}, V::NTuple{2, AbstractArray}, idx
    )
    i, j = idx
    xcorner = grid[1][1][i], grid[2][2][j]
    Vx_grid = V[1][i, j + 1], V[1][i + 1, j + 1]
    Vy_grid = V[2][i + 1, j], V[2][i + 1, j + 1]
    ti = normalize_coordinates(p_i, xcorner, dxi)

    Vx_interp = lerp(ti[1], Vx_grid...)
    Vy_interp = lerp(ti[2], Vy_grid...)

    return Vx_interp, Vy_interp
end

@inline function interp_velocity2particle(
        p_i::Union{SVector, NTuple}, grid::NTuple{3}, dxi::NTuple{3}, V::NTuple{3, AbstractArray}, idx
    )
    i, j, k = idx
    xcorner = grid[1][1][i], grid[2][2][j], grid[3][3][k]
    Vx_grid = V[1][i, j + 1, k + 1], V[1][i + 1, j + 1, k + 1]
    Vy_grid = V[2][i + 1, j, k + 1], V[2][i + 1, j + 1, k + 1]
    Vz_grid = V[3][i + 1, j + 1, k], V[3][i + 1, j + 1, k + 1]
    ti = normalize_coordinates(p_i, xcorner, dxi)

    Vx_interp = lerp(ti[1], Vx_grid...)
    Vy_interp = lerp(ti[2], Vy_grid...)
    Vz_interp = lerp(ti[3], Vz_grid...)

    return Vx_interp, Vy_interp, Vz_interp
end
