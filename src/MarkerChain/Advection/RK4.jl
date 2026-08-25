@inline function advect_particle_markerchain(
        ::RungeKutta4,
        p0::NTuple{N, T},
        V::NTuple{N, AbstractArray{T, N}},
        grid_vi,
        local_limits,
        dt,
        icell;
        backtracking::Bool = false
    ) where {N, T}
    backtracking_sign = 1 - 2 * backtracking # flip sign if backtracking is true, used for backtracking particles during Semi-Lagrangian advection
    k1 = interp_velocity2particle_markerchain(p0, grid_vi, V, icell)
    k2 = interp_velocity2particle_markerchain(p0 .+ backtracking_sign .* dt .* k1 ./ 2, grid_vi, V, icell)
    k3 = interp_velocity2particle_markerchain(p0 .+ backtracking_sign .* dt .* k2 ./ 2, grid_vi, V, icell)
    k4 = interp_velocity2particle_markerchain(p0 .+ backtracking_sign .* dt .* k3, grid_vi, V, icell)

    p = @. p0 + backtracking_sign * dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return p
end
