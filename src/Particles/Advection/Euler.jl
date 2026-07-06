function advect_particle(
        method::Euler,
        p0::NTuple{N, T},
        V::NTuple{N, AbstractArray},
        grid_vi,
        local_limits,
        dxi,
        dt,
        idx::NTuple;
        backtracking::Bool = false
    ) where {N, T}

    # interpolate velocity to current location
    vp0 = interp_velocity2particle(p0, grid_vi, local_limits, dxi, V, idx)

    # first advection stage x = x + v * dt * α
    p1 = first_stage(method, dt, vp0, p0; backtracking = backtracking)

    return p1
end
