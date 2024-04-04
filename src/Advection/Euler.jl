@inline function first_stage(
    ::Euler, dt, particle_velocity::NTuple{N,T}, particle_coordinate::NTuple{N,T}
) where {N,T}
    return @. @muladd particle_coordinate + dt * particle_velocity
end

function advect_particle(
    method::Euler,
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

    return p1
end