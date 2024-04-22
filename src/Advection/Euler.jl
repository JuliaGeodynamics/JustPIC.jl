@inline function first_stage(
    ::Euler, dt, particle_velocity::NTuple{N,T}, particle_coordinate::NTuple{N,T}
) where {N,T}
    return @. @muladd particle_coordinate + dt * particle_velocity
end
