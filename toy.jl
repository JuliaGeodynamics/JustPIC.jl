import JustPIC._2D.lerp

# 2D MQS-x
@inline function MQS(F, v::NTuple{4}, t::NTuple{2}, i, j, ::Val{1})
    t1, t2 = t
    nx, ny = size(F)
    lerp_bot = lerp(v[1:2], (t1,))
    lerp_top = lerp(v[3:4], (t1,))

    v0, v1, v2 = if t1 > 0.5
        F[clamp(i-1, 1, nx), j], v[1], v[2]
    else
        v[1], v[2], F[clamp(i+2, 1, nx), j]
    end
    correction_bot = 0.5 * (t1 - 0.5)^2 * (muladd(-2, v1, v0) + v2)
    # correction_bot = 0.5 * (t1 - 0.5)^2 * (v0 - 2*v1 + v2)
    
    v0, v1, v2 = if t[1] > 0.5
        F[clamp(i-1,1,nx), clamp(j+1, 1, ny)], v[3], v[4]
    else
        v[3], v[4], F[clamp(i+2, 1, nx), clamp(j+1,1,ny)]
    end
    correction_top = 0.5 * (t1 - 0.5)^2 * (muladd(-2, v1, v0) + v2)
    # correction_top = 0.5 * (t1 - 0.5)^2 * (v0 - 2*v1 + v2)

    v0_MQS = lerp_bot + correction_bot * 1
    v1_MQS = lerp_top + correction_top * 1

    return lerp((v0_MQS, v1_MQS), (t2,))
end


# 2D MQS-y
@inline function MQS(F, v::NTuple{4}, t::NTuple{2}, i, j, ::Val{2})
    t1, t2     = t
    nx, ny     = size(F)
    v_left     = (v[1], v[3])
    v_right    = (v[2], v[4])
    lerp_left  = lerp(v_left, (t2,))
    lerp_right = lerp(v_right, (t2,))

    v0, v1, v2 = if t2 < 0.5
        F[i, clamp(j-1, 1, ny)], v_left...
    else
        v_left..., F[i, clamp(j+2, 1, ny)]
    end
    correction_left = 0.5 * (t2 - 0.5)^2 * (muladd(-2, v1, v0) + v2)
    # correction_left = 0.5 * (t2 - 0.5)^2 * (v0 - 2*v1 + v2)

    @show t2
    v0, v1, v2 = if t2 < 0.5
        F[clamp(i+1,1,nx), clamp(j-1, 1, ny)], v_right...
    else
        v_right..., F[clamp(i+1,1,nx), clamp(j+2, 1, ny)]
    end
    correction_right = 0.5 * (t2 - 0.5)^2 * (muladd(-2, v1, v0) + v2)
    # correction_right = 0.5 * (t2 - 0.5)^2 * (v0 - 2*v1 + v2)

    v0_MQS = lerp_left  + correction_left * 0
    v1_MQS = lerp_right + correction_right * 0

    return lerp((v0_MQS, v1_MQS), (t1,))
end