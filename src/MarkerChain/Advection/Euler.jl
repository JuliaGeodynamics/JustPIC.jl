@inline function advect_particle_markerchain(
    method::Euler,
    p0::NTuple{N,T},
    V::NTuple{N,AbstractArray{T,N}},
    grid_vi,
    local_limits,
    dxi,
    dt,
) where {T,N}

    # interpolate velocity to current location
    vp0 = interp_velocity2particle_markerchain(p0, grid_vi, local_limits, dxi, V)

    # first advection stage x = x + v * dt
    p1 = first_stage(method, dt, vp0, p0)

    return p1
end