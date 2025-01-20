function PhaseRatios(
    ::Type{T}, ::Type{B}, nphases::Integer, ni::NTuple{2,Integer}
) where {T,B}
    nx, ny = ni

    center = cell_array(zero(T), (nphases,), ni)
    vertex = cell_array(zero(T), (nphases,), ni .+ 1)
    Vx = cell_array(zero(T), (nphases,), (nx + 1, ny))
    Vy = cell_array(zero(T), (nphases,), (nx, ny + 1))
    dummy = cell_array(zero(T), (nphases,), (1, 1)) # because it cant be a Union{T, Nothing} type on the GPU....

    return JustPIC.PhaseRatios(B, center, vertex, Vx, Vy, dummy, dummy, dummy, dummy)
end

function PhaseRatios(
    ::Type{T}, ::Type{B}, nphases::Integer, ni::NTuple{3,Integer}
) where {T,B}
    nx, ny, nz = ni

    center = cell_array(zero(T), (nphases,), ni)
    vertex = cell_array(zero(T), (nphases,), ni .+ 1)
    Vx = cell_array(zero(T), (nphases,), (nx + 1, ny, nz))
    Vy = cell_array(zero(T), (nphases,), (nx, ny + 1, nz))
    Vz = cell_array(zero(T), (nphases,), (nx, ny, nz + 1))
    yz = cell_array(zero(T), (nphases,), (nx, ny + 1, nz + 1))
    xz = cell_array(zero(T), (nphases,), (nx + 1, ny, nz + 1))
    xy = cell_array(zero(T), (nphases,), (nx + 1, ny + 1, nz))

    return JustPIC.PhaseRatios(B, center, vertex, Vx, Vy, Vz, yz, xz, xy)
end

function PhaseRatios(nphases::Integer, ni::NTuple{N,Integer}) where {N}
    return PhaseRatios(Float64, CPUBackend, nphases, ni)
end

function PhaseRatios(
    ::Type{B}, nphases::Integer, ni::NTuple{N,Integer}
) where {N,B<:AbstractBackend}
    return PhaseRatios(Float64, B, nphases, ni)
end
