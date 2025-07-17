@inline function advect_particle(
        method::RungeKutta2,
        p0::NTuple{N},
        V::NTuple{N},
        grid_vi,
        local_limits,
        dxi,
        dt,
        I::NTuple{N};
        backtracking::Bool = false
    ) where {N}

    # interpolate velocity to current location
    vp0 = interp_velocity2particle(p0, grid_vi, local_limits, dxi, V, I)

    # first advection stage x = x + v * dt * α
    p1 = first_stage(method, dt, vp0, p0; backtracking = backtracking)

    # interpolate velocity to new location
    vp1 = interp_velocity2particle(p1, grid_vi, local_limits, dxi, V, I)

    # final advection step
    p2 = second_stage(method, dt, vp0, vp1, p0; backtracking = backtracking)

    return p2
end

@inline function advect_particle(
        method::RungeKutta2,
        p0::NTuple{N},
        V::NTuple{N},
        grid_vi,
        local_limits,
        dxi,
        dt,
        interpolation_fn::F,
        I::NTuple;
        backtracking::Bool = false
    ) where {N, F}

    # interpolate velocity to current location
    vp0 = interpolation_fn(p0, grid_vi, local_limits, dxi, V, I)

    # first advection stage x = x + v * dt * α
    p1 = first_stage(method, dt, vp0, p0; backtracking = backtracking)

    # interpolate velocity to new location
    vp1 = interpolation_fn(p1, grid_vi, local_limits, dxi, V, I)

    # final advection step
    p2 = second_stage(method, dt, vp0, vp1, p0; backtracking = backtracking)

    return p2
end

@inline function advect_particle_SML(
        method::RungeKutta2,
        p0::NTuple{N},
        V::NTuple{N},
        grid_vi,
        dxi,
        dt,
        I::NTuple{N};
        backtracking::Bool = false
    ) where {N}

    # interpolate velocity to current location
    vp0 = interp_velocity2particle(p0, grid_vi, dxi, V, I)

    # first advection stage x = x + v * dt * α
    p1 = first_stage(method, dt, vp0, p0; backtracking = backtracking)

    # interpolate velocity to new location
    vp1 = interp_velocity2particle(p1, grid_vi, dxi, V, I)

    # final advection step
    p2 = second_stage(method, dt, vp0, vp1, p0; backtracking = backtracking)

    return p2
end

@inline function advect_particle_SML(
        method::RungeKutta2,
        p0::NTuple{N},
        V::NTuple{N},
        grid_vi,
        dxi,
        dt,
        interpolation_fn::F,
        I::NTuple;
        backtracking::Bool = false
    ) where {N, F}

    # interpolate velocity to current location
    vp0 = interpolation_fn(p0, grid_vi, dxi, V, I)

    # first advection stage x = x + v * dt * α
    p1 = first_stage(method, dt, vp0, p0; backtracking = backtracking)

    # interpolate velocity to new location
    vp1 = interpolation_fn(p1, grid_vi, dxi, V, I)

    # final advection step
    p2 = second_stage(method, dt, vp0, vp1, p0; backtracking = backtracking)

    return p2
end
