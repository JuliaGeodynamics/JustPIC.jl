@inline function first_stage(
        integrator::RungeKutta2,
        dt,
        particle_velocity::NTuple{N, T},
        particle_coordinate::NTuple{N, T};
        backtracking::Bool = false
    ) where {N, T}
    (; α) = integrator
    backtracking_sign = 1 - 2 * backtracking # flip sign if backtracking is true, used for backtracking particles during Semi-Lagrangian advection
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
    (; α) = integrator
    backtracking_sign = 1 - 2 * backtracking # flip sign if backtracking is true, used for backtracking particles during Semi-Lagrangian advection
    p = if α == 0.5
        @. @muladd particle_coordinate + backtracking_sign * dt * particle_velocity1
    else
        @. @muladd particle_coordinate +
            backtracking_sign * dt * (
            (1.0 - 0.5 * inv(α)) * particle_velocity0 + 0.5 * inv(α) * particle_velocity1
        )
    end
    return p
end
