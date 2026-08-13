@inline function advect_particle(
        ::RungeKutta4,
        p0::NTuple{N},
        V::NTuple{N},
        grid_vi,
        local_limits,
        dxi,
        dt,
        idx::NTuple{N},
        periodicity,
        domain_limits,
    ) where {N}

    k1 = interp_velocity2particle(p0, grid_vi, local_limits, dxi, V, idx)
    p1 = wrap_position(p0 .+ dt .* k1 ./ 2, periodicity, domain_limits)
    k2 = interp_velocity2particle(p1, grid_vi, local_limits, dxi, V, idx)
    p2 = wrap_position(p0 .+ dt .* k2 ./ 2, periodicity, domain_limits)
    k3 = interp_velocity2particle(p2, grid_vi, local_limits, dxi, V, idx)
    p3 = wrap_position(p0 .+ dt .* k3, periodicity, domain_limits)
    k4 = interp_velocity2particle(p3, grid_vi, local_limits, dxi, V, idx)

    p = @. p0 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return wrap_position(p, periodicity, domain_limits)
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
        periodicity,
        domain_limits,
    ) where {N, F}

    k1 = interpolation_fn(p0, grid_vi, local_limits, dxi, V, idx)
    p1 = wrap_position(p0 .+ dt .* k1 ./ 2, periodicity, domain_limits)
    k2 = interpolation_fn(p1, grid_vi, local_limits, dxi, V, idx)
    p2 = wrap_position(p0 .+ dt .* k2 ./ 2, periodicity, domain_limits)
    k3 = interpolation_fn(p2, grid_vi, local_limits, dxi, V, idx)
    p3 = wrap_position(p0 .+ dt .* k3, periodicity, domain_limits)
    k4 = interpolation_fn(p3, grid_vi, local_limits, dxi, V, idx)

    p = @. p0 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return wrap_position(p, periodicity, domain_limits)
end


@inline function advect_particle_SML(
        ::RungeKutta4,
        p0::NTuple{N},
        V::NTuple{N},
        grid_vi,
        dxi,
        dt,
        idx::NTuple{N};
        backtracking::Bool = false
    ) where {N}

    backtracking_sign = 1 - 2 * backtracking # flip sign if backtracking is true, used for backtracking particles during Semi-Lagrangian advection
    k1 = interp_velocity2particle(p0, grid_vi, dxi, V, idx)
    k2 = interp_velocity2particle(p0 .+ backtracking_sign .* dt .* k1 ./ 2, grid_vi, dxi, V, idx)
    k3 = interp_velocity2particle(p0 .+ backtracking_sign .* dt .* k2 ./ 2, grid_vi, dxi, V, idx)
    k4 = interp_velocity2particle(p0 .+ backtracking_sign .* dt .* k3, grid_vi, dxi, V, idx)

    p = @. p0 + backtracking_sign * dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return p
end

@inline function advect_particle_SML(
        ::RungeKutta4,
        p0::NTuple{N},
        V::NTuple{N},
        grid_vi,
        dxi,
        dt,
        interpolation_fn::F,
        idx::NTuple;
        backtracking::Bool = false
    ) where {N, F}

    backtracking_sign = 1 - 2 * backtracking # flip sign if backtracking is true, used for backtracking particles during Semi-Lagrangian advection
    k1 = interpolation_fn(p0, grid_vi, dxi, V, idx)
    k2 = interpolation_fn(p0 .+ backtracking_sign .* dt .* k1 ./ 2, grid_vi, dxi, V, idx)
    k3 = interpolation_fn(p0 .+ backtracking_sign .* dt .* k2 ./ 2, grid_vi, dxi, V, idx)
    k4 = interpolation_fn(p0 .+ backtracking_sign .* dt .* k3, grid_vi, dxi, V, idx)

    p = @. p0 + backtracking_sign * dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return p
end
