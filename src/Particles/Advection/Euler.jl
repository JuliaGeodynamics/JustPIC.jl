function advect_particle(
        method::Euler,
        p0::NTuple{N, T},
        V::NTuple{N, AbstractArray},
        grid_vi,
        local_limits,
        dxi,
        dt,
        idx::NTuple,
        periodicity,
        domain_limits;
        backtracking::Bool = false
    ) where {N, T}

    # interpolate velocity to current location
    vp0 = interp_velocity2particle(p0, grid_vi, local_limits, dxi, V, idx)

    # first advection stage x = x + v * dt * α
    p1 = first_stage(method, dt, vp0, p0; backtracking = backtracking)

    return wrap_position(p1, periodicity, domain_limits)
end

@inline function advect_particle(
        method::Euler,
        p0::NTuple{N},
        V::NTuple{N},
        grid_vi,
        local_limits,
        dxi,
        dt,
        interpolation_fn::F,
        idx::NTuple,
        periodicity,
        domain_limits;
        backtracking::Bool = false
    ) where {N, F}
    vp0 = interpolation_fn(p0, grid_vi, local_limits, dxi, V, idx)
    p1 = first_stage(method, dt, vp0, p0; backtracking = backtracking)
    return wrap_position(p1, periodicity, domain_limits)
end
