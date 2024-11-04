function PhaseRatios(
    ::Type{T}, ::Type{B}, nphases::Integer, ni::NTuple{2,Integer}
) where {T,B}
    center = cell_array(0.0, (nphases,), ni)
    vertex = cell_array(0.0, (nphases,), ni .+ 1)

    nx, ny = ni
    Vx = cell_array(0.0, (nphases,), nx + 1, ny)
    Vy = cell_array(0.0, (nphases,), nx, ny + 1)
    dummy = cell_array(0.0, (nphases,), 1, 1) # because it cant be a Union{T, Nothing} type on the GPU....
   
    return JustPIC.PhaseRatios(B, center, vertex, Vx, Vy, dummy, dummy, dummy, dummy)
end

function PhaseRatios(
    ::Type{T}, ::Type{B}, nphases::Integer, ni::NTuple{3,Integer}
) where {T,B}
    center = cell_array(0.0, (nphases,), ni)
    vertex = cell_array(0.0, (nphases,), ni .+ 1)

    nx, ny, nz  = ni
    Vx = cell_array(0.0, (nphases,), nx + 1, ny, nz)
    Vy = cell_array(0.0, (nphases,), nx, ny + 1, nz)
    Vz = cell_array(0.0, (nphases,), nx, ny, nz + 1)
    yz = cell_array(0.0, (nphases,), nx, ny + 1, nz + 1)
    xz = cell_array(0.0, (nphases,), nx + 1, ny, nz + 1)
    xy = cell_array(0.0, (nphases,), nx + 1, ny + 1, nz)
   
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
