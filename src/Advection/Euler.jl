@inline function first_stage(
        ::Euler, dt, particle_velocity::NTuple{N, T}, particle_coordinate::NTuple{N, T}; backtracking::Bool = false
    ) where {N, T}
    backtracking_sign = 1 - 2 * backtracking # flip sign if backtracking is true, used for backtracking particles during Semi-Lagrangian advection
    # return @. @muladd particle_coordinate + backtracking_sign * dt * particle_velocity
    return @. @muladd particle_coordinate + dt * particle_velocity
end