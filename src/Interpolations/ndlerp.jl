"""
    lerp(t, v0, v1)

Linearly interpolates between `v0` and `v1` using the parameter `t`.

# Arguments
- `t`: The interpolation parameter.
- `v0`: The starting value.
- `v1`: The ending value.

"""
@inline lerp(t, v0, v1) = fma(t, v1, fma(-t, v0, v0))

"""
    lerp(v, t::NTuple{nD,T}) where {nD,T}

Linearly interpolates the value `v` between the elements of the tuple `t`.
This function is specialized for tuples of length `nD`.

# Arguments
- `v`: The value to be interpolated.
- `t`: The tuple of values to interpolate between.
"""
@inline lerp(v, t::NTuple{nD,T})              where {nD,T}   = lerp(v, t, 0, Val(nD - 1))
@inline lerp(v, t::NTuple{nD,T}, i, ::Val{N}) where {nD,N,T} = lerp(t[N + 1], lerp(v, t, i, Val(N - 1)), lerp(v, t, i + 2^N, Val(N - 1)))
@inline lerp(v, t::NTuple{nD,T}, i, ::Val{0}) where {nD,T}   = lerp(t[1], v[i + 1], v[i + 2])
