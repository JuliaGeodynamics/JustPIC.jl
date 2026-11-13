@inline function advect_particle_markerchain(
        ::RungeKutta4,
        p0::NTuple{N, Any},
        V::NTuple{N, AbstractArray{<:Any, N}},
        grid_vi,
        local_limits,
        dt,
        icell;
        backtracking::Bool = false,
        periodic::Bool = false,
        xlims = (p0[1], p0[1]),
    ) where {N}
    backtracking_sign = 1 - 2 * backtracking # flip sign if backtracking is true, used for backtracking particles during Semi-Lagrangian advection
    k1 = interp_velocity2particle_markerchain(p0, grid_vi, V, icell)
    half_step = backtracking_sign * dt / 2
    p1 = ntuple(Val(N)) do i
        p0[i] + half_step * k1[i]
    end
    k2 = interp_velocity2particle_markerchain(_wrap_stage(p1, periodic, xlims), grid_vi, V, icell)
    p2 = ntuple(Val(N)) do i
        p0[i] + half_step * k2[i]
    end
    k3 = interp_velocity2particle_markerchain(_wrap_stage(p2, periodic, xlims), grid_vi, V, icell)
    p3 = ntuple(Val(N)) do i
        p0[i] + backtracking_sign * dt * k3[i]
    end
    k4 = interp_velocity2particle_markerchain(_wrap_stage(p3, periodic, xlims), grid_vi, V, icell)

    p = ntuple(Val(N)) do i
        p0[i] + backtracking_sign * dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6
    end
    return p
end
