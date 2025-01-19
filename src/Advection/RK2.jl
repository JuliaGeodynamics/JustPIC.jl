@inline function first_stage(
        integrator::RungeKutta2,
        dt,
        particle_velocity::NTuple{N, T},
        particle_coordinate::NTuple{N, T},
    ) where {N, T}
    (; α) = integrator
    return @. @muladd particle_coordinate + dt * α * particle_velocity
end

@inline function second_stage(
        integrator::RungeKutta2,
        dt,
        particle_velocity0::NTuple{N, T},
        particle_velocity1::NTuple{N, T},
        particle_coordinate::NTuple{N, T},
    ) where {N, T}
    (; α) = integrator
    p = if α == 0.5
        @. @muladd particle_coordinate + dt * particle_velocity1
    else
        @. @muladd particle_coordinate +
            dt * (
            (1.0 - 0.5 * inv(α)) * particle_velocity0 + 0.5 * inv(α) * particle_velocity1
        )
    end
    return p
end
