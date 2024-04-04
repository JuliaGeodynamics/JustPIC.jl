@inline function advect_particle(
    method::RungeKutta2,
    p0::NTuple{N,T},
    V::NTuple{N,AbstractArray{T,N}},
    grid_vi,
    local_limits,
    dxi,
    dt,
    idx::NTuple,
) where {T,N}

    # interpolate velocity to current location
    vp0 = interp_velocity2particle(p0, grid_vi, local_limits, dxi, V, idx)

    # first advection stage x = x + v * dt * α
    p1 = first_stage(method, dt, vp0, p0)

    # interpolate velocity to new location
    vp1 = interp_velocity2particle(p1, grid_vi, local_limits, dxi, V, idx)

    # final advection step
    p2 = second_stage(method, dt, vp0, vp1, p0)

    return p2
end

@inline function first_stage(
    integrator::RungeKutta2, dt, particle_velocity::NTuple{N,T}, particle_coordinate::NTuple{N,T}
) where {N,T}
    (; α) = integrator
    return @. @muladd particle_coordinate + dt * α * particle_velocity
end

@inline function second_stage(
    integrator::RungeKutta2,
    dt,
    particle_velocity0::NTuple{N,T},
    particle_velocity1::NTuple{N,T},
    particle_coordinate::NTuple{N,T},
) where {N,T}
    (; α) = integrator
    p = if α == 0.5
        @. @muladd particle_coordinate + dt * particle_velocity1
    else
        @. @muladd particle_coordinate +
        dt * (
            (1.0 - 0.5 * inv(α)) * particle_velocity0 + 0.5 * inv(α) * particle_velocity1
        )
    end
    return p
end