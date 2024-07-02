"""
    lerp(v, t::NTuple{nD,T}) where {nD,T}

Linearly interpolates the value `v` between the elements of the tuple `t`.
This function is specialized for tuples of length `nD`.

# Arguments
- `v`: The value to be interpolated.
- `t`: The tuple of values to interpolate between.
"""
@inline lerp(v, t::NTuple{nD,T})              where {nD,T}    = lerp(v, t, 0, Val(nD - 1))
@inline lerp(v, t::NTuple{nD,T}, i, ::Val{N}) where {nD,N,T}  = lerp(t[N + 1], lerp(v, t, i, Val(N - 1)), lerp(v, t, i + 2^N, Val(N - 1)))
@inline lerp(v, t::NTuple{nD,T}, i, ::Val{0}) where {nD,T}    = lerp(t[1], v[i + 1], v[i + 2])
@inline lerp(t::T, v0::T, v1::T)              where {T<:Real} = fma(t, v1, fma(-t, v0, v0))

# """
#     MQS(v, t::NTuple{nD,T}) where {nD,T}

# Modified Quadratic Spline interpolation of the value `v` between the elements of the tuple `t`.
# This function is specialized for tuples of length `nD`.

# # Arguments
# - `v`: The value to be interpolated.
# - `t`: The tuple of values to interpolate between.
# """
# @inline MQS(v, t::NTuple{nD,T})              where {nD,T}    = MQS(v, t, 0, Val(nD - 1))
# @inline MQS(v, t::NTuple{nD,T}, i, ::Val{N}) where {nD,N,T}  = lerp(t[N + 1], MQS(v, t, i, Val(N - 1)), MQS(v, t, i + 2^N, Val(N - 1)))
# @inline MQS(v, t::NTuple{nD,T}, i, ::Val{0}) where {nD,T}    = MQS(t[1], v[i + 1], v[i + 2], v[i + 3])
# @inline function MQS(t::T, v0::T, v1::T, v2::T) where {T<:Real} 
#     linear_term = muladd(t, v2, muladd(-t, v1, v1))
#     quadratic_correction = 0.5 * (t - 1.5)^2 * (muladd(-2, v1, v0) + v2)
#     return linear_term + quadratic_correction
# end

@inline function MQS(t::T, v0::T, v1::T, v2::T) where {T<:Real} 
    linear_term = muladd(t, v2, muladd(-t, v1, v1))
    quadratic_correction = 0.5 * (t - 1.5)^2 * (muladd(-2, v1, v0) + v2)
    return linear_term + quadratic_correction
end

function MQS_Vx(v_cell, v_MQS, t::NTuple{2})
    # MQS for Vx - bottom
    v_bot = MQS(t[1], v_MQS[1], v_cell[1:2]...)
    # MQS for Vx - top
    v_top = MQS(t[1], v_MQS[2], v_cell[3:4]...)

    vt = lerp(t[2], v_bot, v_top)
end

function MQS_Vy(v_cell, v_MQS, t::NTuple{2})
    # MQS for Vy - left
    v_left = MQS(t[2], v_MQS[1], v_cell[1], v_cell[3])
    # MQS for Vy - right
    v_right = MQS(t[2], v_MQS[2], v_cell[2], v_cell[4])

    vt = lerp(t[1], v_left, v_right)
end