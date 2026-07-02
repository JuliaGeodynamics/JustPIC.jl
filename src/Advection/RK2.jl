@inline function first_stage(
        integrator::RungeKutta2,
        dt,
        particle_velocity::NTuple{N, T},
        particle_coordinate::NTuple{N, T};
        backtracking::Bool = false
    ) where {N, T}
    # work in the particle-coordinate precision T so Float32 backends (e.g. Metal,
    # which has no Float64) are not silently promoted by a Float64 α / dt / literal.
    α = convert(T, integrator.α)
    dt = convert(T, dt)
    backtracking_sign = convert(T, 1 - 2 * backtracking) # flip sign if backtracking is true, used for backtracking particles during Semi-Lagrangian advection
    return @. @muladd particle_coordinate + backtracking_sign * α * dt * particle_velocity
end

@inline function second_stage(
        integrator::RungeKutta2,
        dt,
        particle_velocity0::NTuple{N, T},
        particle_velocity1::NTuple{N, T},
        particle_coordinate::NTuple{N, T};
        backtracking::Bool = false
    ) where {N, T}
    # work in the particle-coordinate precision T (see first_stage).
    α = convert(T, integrator.α)
    dt = convert(T, dt)
    backtracking_sign = convert(T, 1 - 2 * backtracking) # flip sign if backtracking is true, used for backtracking particles during Semi-Lagrangian advection
    half = convert(T, 0.5)
    p = if α == half
        @. @muladd particle_coordinate + backtracking_sign * dt * particle_velocity1
    else
        @. @muladd particle_coordinate +
            backtracking_sign * dt * (
            (one(T) - half * inv(α)) * particle_velocity0 + half * inv(α) * particle_velocity1
        )
    end
    return p
end
