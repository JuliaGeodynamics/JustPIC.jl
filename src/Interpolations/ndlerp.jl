"""
    lerp(v, t::NTuple{nD,T}) where {nD,T}

Linearly interpolates the value `v` between the elements of the tuple `t`.
This function is specialized for tuples of length `nD`.

# Arguments
- `v`: The value to be interpolated.
- `t`: The tuple of values to interpolate between.
"""
@inline lerp(v, t::NTuple{nD, T}) where {nD, T} = lerp(v, t, 0, Val(nD - 1))
@inline lerp(v, t::NTuple{nD, T}, i, ::Val{N}) where {nD, N, T} =
    lerp(t[N + 1], lerp(v, t, i, Val(N - 1)), lerp(v, t, i + 2^N, Val(N - 1)))
@inline lerp(v, t::NTuple{nD, T}, i, ::Val{0}) where {nD, T} = lerp(t[1], v[i + 1], v[i + 2])
@inline lerp(t::T, v0::T, v1::T) where {T <: Real} = muladd(t, v1, muladd(-t, v0, v0))
