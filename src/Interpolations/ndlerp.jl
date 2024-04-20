@inline lerp(t, v0, v1) = fma(t, v1, fma(-t, v0, v0))

@inline lerp(v, t::NTuple{nD,T}) where {nD,T} = lerp(v, t, 0, Val(nD - 1))
@inline lerp(v, t::NTuple{nD,T}, i, ::Val{N}) where {nD,N,T} =
    lerp(t[N + 1], lerp(v, t, i, Val(N - 1)), lerp(v, t, i + 2^N, Val(N - 1)))
@inline lerp(v, t::NTuple{nD,T}, i, ::Val{0}) where {nD,T} =
    lerp(t[1], v[i + 1], v[i + 2])
