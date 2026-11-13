@inline function advect_particle_markerchain(
        method::RungeKutta2,
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

    # interpolate velocity to current location
    vp0 = interp_velocity2particle_markerchain(p0, grid_vi, V, icell)

    # first advection stage x = x + v * dt * α
    p1 = first_stage(method, dt, vp0, p0; backtracking = backtracking)

    # interpolate velocity to new location
    vp1 = interp_velocity2particle_markerchain(_wrap_stage(p1, periodic, xlims), grid_vi, V, icell)

    # final advection step
    p2 = second_stage(method, dt, vp0, vp1, p0; backtracking = backtracking)

    return p2
end

# An intermediate stage of a periodic trajectory is wrapped into `xlims` before the velocity is
# sampled there; the final position is not, so the caller sees a continuous displacement.
@inline _wrap_stage(p, periodic, xlims) = periodic ? (wrap_coordinate(p[1], true, xlims), Base.tail(p)...) : p
