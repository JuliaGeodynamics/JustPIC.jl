@inline function advect_particle(
        ::RungeKutta4,
        p0::NTuple{N},
        V::NTuple{N},
        grid_vi,
        local_limits,
        dxi,
        dt,
        idx::NTuple{N},
    ) where {N}
    # interpolate velocity to current location
    v1 = interp_velocity2particle(p0, grid_vi, local_limits, dxi, V, idx)
    k1 = @. dt * v1

    # second stage
    v2 = interp_velocity2particle(p0 .+ k1 ./ 2, grid_vi, local_limits, dxi, V, idx)
    k2 = @. dt * v2 / 2

    v3 = interp_velocity2particle(p0 .+ k2 ./ 2, grid_vi, local_limits, dxi, V, idx)
    k3 = @. dt * v3 / 2

    v4 = interp_velocity2particle(p0 .+ k3, grid_vi, local_limits, dxi, V, idx)
    k4 = @. dt * v4

    p = @. p0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return p
end

@inline function advect_particle(
        ::RungeKutta4,
        p0::NTuple{N},
        V::NTuple{N},
        grid_vi,
        local_limits,
        dxi,
        dt,
        interpolation_fn::F,
        idx::NTuple,
    ) where {N, F}

    # interpolate velocity to current location
    v1 = interpolation_fn(p0, grid_vi, local_limits, dxi, V, idx)
    k1 = @. dt * v1

    # second stage
    v2 = interpolation_fn(p0 .+ k1 ./ 2, grid_vi, local_limits, dxi, V, idx)
    k2 = @. dt * v2 / 2

    v3 = interpolation_fn(p0 .+ k2 ./ 2, grid_vi, local_limits, dxi, V, idx)
    k3 = @. dt * v3 / 2

    v4 = interpolation_fn(p0 .+ k3, grid_vi, local_limits, dxi, V, idx)
    k4 = @. dt * v4

    p = @. p0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return p
end
