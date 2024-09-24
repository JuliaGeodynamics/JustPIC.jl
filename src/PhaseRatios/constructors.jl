function PhaseRatios(
    ::Type{T}, ::Type{B}, nphases::Integer, ni::NTuple{N,Integer}
) where {N,T,B}
    center = cell_array(0.0, (nphases,), ni)
    vertex = cell_array(0.0, (nphases,), ni .+ 1)

    return JustPIC.PhaseRatios{B,typeof(center)}(center, vertex)
end

function PhaseRatios(nphases::Integer, ni::NTuple{N,Integer}) where {N}
    return PhaseRatios(Float64, CPUBackend, nphases, ni)
end

function PhaseRatios(
    ::Type{B}, nphases::Integer, ni::NTuple{N,Integer}
) where {N,B<:AbstractBackend}
    return PhaseRatios(Float64, B, nphases, ni)
end
