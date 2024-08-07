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
@inline MQS(v, t::NTuple{nD,T})              where {nD,T}    = MQS(v, t, 0, Val(nD - 1))
@inline MQS(v, t::NTuple{nD,T}, i, ::Val{N}) where {nD,N,T}  = lerp(t[N + 1], MQS(v, t, i, Val(N - 1)), MQS(v, t, i + 2^N + 1, Val(N - 1)))
@inline MQS(v, t::NTuple{nD,T}, i, ::Val{0}) where {nD,T}    = MQS(t[1], v[i + 1], v[i + 2], v[i + 3])
@inline function MQS(t::T, v0::T, v1::T, v2::T) where {T<:Real} 
    linear_term = muladd(t, v2, muladd(-t, v1, v1))
    quadratic_correction = 0.5 * (t - 1.5)^2 * (muladd(-2, v1, v0) + v2)
    # @show linear_term
    # @show quadratic_correction 
    return linear_term + quadratic_correction
end

function MQS(v::NTuple{12}, t::NTuple{3})
    front_bot_mqs = MQS(t[1], v[1:3]...)
    front_top_mqs = MQS(t[1], v[4:6]...)
    back_bot_mqs  = MQS(t[1], v[7:9]...)
    back_top_mqs  = MQS(t[1], v[10:end]...)    
    v = front_bot_mqs, back_bot_mqs, front_top_mqs, back_top_mqs
    lerp(v, t[2:3])
end

####
using Test
using JustPIC, JustPIC._2D
import JustPIC._2D: lerp, MQS

@testset "Interpolation kernels" begin
    @testset "lerp" begin
        t1D = (0.5, )
        v1D = 1e0, 2e0

        @test lerp(v1D, t1D) == 1.5

        t2D = 0.5, 0.5 
        v2D = 1e0, 2e0, 1e0, 2e0
        @test lerp(v2D, t2D)  == 1.5

        t3D = 0.5, 0.5, 0.5 
        v3D = 1e0, 2e0, 1e0, 2e0, 1e0, 2e0, 1e0, 2e0
        @test lerp(v3D, t3D) == 1.5
    end

    @testset "MQS" begin
        t2D = 0.5, 0.5 
        v2D = 0e0, 1e0, 2e0, 0e0, 1e0, 2e0
        @test MQS(v2D, t2D) == 1.5

        bot_mqs = MQS(t2D[1], v2D[1:3]...)
        top_mqs = MQS(t2D[1], v2D[4:6]...)
        @test lerp(t2D[1], bot_mqs, top_mqs) == mqs

        t3D = 0.5, 0.5, 0.5 
        v3D = (
            0e0, 1e0, 2e0,
            0e0, 1e0, 2e0,
            0e0, 1e0, 2e0,
            0e0, 1e0, 2e0,
        )

        @test MQS(v3D, t3D) == 1.5
    end
end