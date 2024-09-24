"""
    lerp(v, t::NTuple{nD,T}) where {nD,T}

Linearly interpolates the value `v` between the elements of the tuple `t`.
This function is specialized for tuples of length `nD`.

# Arguments
- `v`: The value to be interpolated.
- `t`: The tuple of values to interpolate between.
"""
@inline lerp(v, t::NTuple{nD,T}) where {nD,T} = lerp(v, t, 0, Val(nD - 1))
@inline lerp(v, t::NTuple{nD,T}, i, ::Val{N}) where {nD,N,T} =
    lerp(t[N + 1], lerp(v, t, i, Val(N - 1)), lerp(v, t, i + 2^N, Val(N - 1)))
@inline lerp(v, t::NTuple{nD,T}, i, ::Val{0}) where {nD,T} = lerp(t[1], v[i + 1], v[i + 2])
@inline lerp(t::T, v0::T, v1::T) where {T<:Real} = muladd(t, v1, muladd(-t, v0, v0))

"""
    MQS(v, t::NTuple{nD,T}) where {nD,T}

Modified Quadratic Spline interpolation of the value `v` between the elements of the tuple `t`.
This function is specialized for tuples of length `nD`.

# Arguments
- `v`: The value to be interpolated.
- `t`: The tuple of values to interpolate between.
"""
# @inline MQS(v, t::NTuple{nD,T})              where {nD,T}    = MQS(v, t, 0, Val(nD - 1))
# @inline MQS(v, t::NTuple{nD,T}, i, ::Val{N}) where {nD,N,T}  = lerp(t[N + 1], MQS(v, t, i, Val(N - 1)), MQS(v, t, i + 2^N + 1, Val(N - 1)))
# @inline MQS(v, t::NTuple{nD,T}, i, ::Val{0}) where {nD,T}    = MQS(t[1], v[i + 1], v[i + 2], v[i + 3])
# @inline function MQS(t::T, v0::T, v1::T, v2::T) where {T<:Real} 
#     linear_term = lerp(t, v1, v2)
#     quadratic_correction = 0.5 * (t - 0.5)^2 * (muladd(-2, v1, v0) + v2)
#     return linear_term + quadratic_correction
# end

# @inline function MQS(v::NTuple{12}, t::NTuple{3})
#     front_bot_mqs = MQS(t[1], v[1:3]...)
#     front_top_mqs = MQS(t[1], v[4:6]...)
#     back_bot_mqs  = MQS(t[1], v[7:9]...)
#     back_top_mqs  = MQS(t[1], v[10:end]...)
#     v = front_bot_mqs, back_bot_mqs, front_top_mqs, back_top_mqs
#     return lerp(v, (t[2], t[3]))
# end

# @inline function MQS(F, v::NTuple{4}, t::NTuple{2}, i, j, ::Val{1})
#     t1, t2 = t
#     nx, ny = size(F)
#     lerp_bot = lerp(v[1:2], (t1,))
#     lerp_top = lerp(v[3:4], (t1,))

#     v0, v1, v2 = if t1 < 0.5
#         F[i-1, j], v[1], v[2]
#     else
#         v[1], v[2], F[i+2, j]
#     end
#     correction_bot = 0.5 * (t1 - 0.5)^2 * (muladd(-2, v1, v0) + v2)
#     # correction_bot = 0.5 * (t1 - 0.5)^2 * (v0 - 2*v1 + v2)

#     v0, v1, v2 = if t[1] < 0.5
#         F[i-1, j+1], v[3], v[4]
#     else
#         v[3], v[4], F[i+2, j+1]
#     end
#     correction_top = 0.5 * (t1 - 0.5)^2 * (muladd(-2, v1, v0) + v2)
#     # correction_top = 0.5 * (t1 - 0.5)^2 * (v0 - 2*v1 + v2)

#     v0_MQS = lerp_bot + correction_bot * 1
#     v1_MQS = lerp_top + correction_top * 1

#     return lerp((v0_MQS, v1_MQS), (t2,))
# end

# # 2D MQS-y
# @inline function MQS(F, v::NTuple{4}, t::NTuple{2}, i, j, ::Val{2})
#     t1, t2     = t
#     nx, ny     = size(F)
#     v_left     = (v[1], v[3])
#     v_right    = (v[2], v[4])
#     lerp_left  = lerp(v_left, (t2,))
#     lerp_right = lerp(v_right, (t2,))

#     v0, v1, v2 = if t2 < 0.5
#         F[i, j-1], v_left...
#     else
#         v_left..., F[i, j+2]
#     end
#     correction_left = 0.5 * (t2 - 0.5)^2 * (muladd(-2, v1, v0) + v2)

#     v0, v1, v2 = if t2 < 0.5
#         F[i+1, j-1], v_right...
#     else
#         v_right..., F[i+1, j+2]
#     end
#     correction_right = 0.5 * (t2 - 0.5)^2 * (muladd(-2, v1, v0) + v2)

#     v0_MQS = lerp_left  + correction_left
#     v1_MQS = lerp_right + correction_right

#     return lerp((v0_MQS, v1_MQS), (t1,))
# end
