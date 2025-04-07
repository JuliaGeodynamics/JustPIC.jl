@inline function advect_particle_markerchain(
        ::RungeKutta4,
        p0::NTuple{N, T},
        V::NTuple{N, AbstractArray{T, N}},
        grid_vi,
        local_limits,
        dxi,
        dt,
    ) where {N, T}
    # interpolate velocity to current location
    v1 = interp_velocity2particle_markerchain(p0, grid_vi, local_limits, dxi, V)
    k1 = @. dt * v1

    # second stage
    v2 = interp_velocity2particle_markerchain(p0 .+ k1 ./ 2, grid_vi, local_limits, dxi, V)
    k2 = @. dt * v2 / 2

    v3 = interp_velocity2particle_markerchain(p0 .+ k2 ./ 2, grid_vi, local_limits, dxi, V)
    k3 = @. dt * v3 / 2

    v4 = interp_velocity2particle_markerchain(p0 .+ k3, grid_vi, local_limits, dxi, V)
    k4 = @. dt * v4

    p = @. p0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return p
end
