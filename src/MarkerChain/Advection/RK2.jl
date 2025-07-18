@inline function advect_particle_markerchain(
        method::RungeKutta2,
        p0::NTuple{N, T},
        V::NTuple{N, AbstractArray{T, N}},
        grid_vi,
        local_limits,
        dxi,
        dt;
        backtracking::Bool=false
    ) where {T, N}

    # interpolate velocity to current location
    vp0 = interp_velocity2particle_markerchain(p0, grid_vi, dxi, V)

    # first advection stage x = x + v * dt * α
    p1 = first_stage(method, dt, vp0, p0; backtracking = backtracking)

    # interpolate velocity to new location
    vp1 = interp_velocity2particle_markerchain(p1, grid_vi, dxi, V)

    # final advection step
    p2 = second_stage(method, dt, vp0, vp1, p0; backtracking = backtracking)

    return p2
end
