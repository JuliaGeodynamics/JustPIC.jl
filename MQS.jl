@inline function MQS(v::NTuple{12}, t::NTuple{3})
    front_bot_mqs = MQS(t[1], v[1:3]...)
    front_top_mqs = MQS(t[1], v[4:6]...)
    back_bot_mqs  = MQS(t[1], v[7:9]...)
    back_top_mqs  = MQS(t[1], v[10:end]...)
    v = front_bot_mqs, back_bot_mqs, front_top_mqs, back_top_mqs
    return lerp(v, (t[2], t[3]))
end


# 2D MQS-x
@inline function MQS(v::NTuple{4}, t::NTuple{2}, i, j, ::Val{1})
    t1, t2   = t
    nx, ny   = size(F)
    v_bot    = v[1], v[2]
    v_top    = v[3], v[4]
    lerp_bot = lerp(v[1:2], (t1,))
    lerp_top = lerp(v[3:4], (t1,))

    correction_bot, correction_top = 0e0, 0e0

    if 2 < i < nx-1  && j < ny # do lerp on the boundaries
        v0, v1, v2 = if t1 < 0.5
            F[i-1, j], v_bot...
        else
            v_bot..., F[i+2, j]
        end
        correction_bot = 0.5 * (t1 - 0.5)^2 * (muladd(-2, v1, v0) + v2)
        
        v0, v1, v2 = if t[1] < 0.5
            F[i-1, j+1], v_top...
        else
            v_top..., F[i+2, j+1]
        end
        correction_top = 0.5 * (t1 - 0.5)^2 * (muladd(-2, v1, v0) + v2)
    end

    v0_MQS = lerp_bot + correction_bot
    v1_MQS = lerp_top + correction_top

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

    correction_left, correction_right = 0e0, 0e0
    if 2 < j < ny && i < nx # do lerp on the boundaries
        v0, v1, v2 = if t2 < 0.5
            F[i, j-1], v_left...
        else
            v_left..., F[i, j+2]
        end
        correction_left = 0.5 * (t2 - 0.5)^2 * (muladd(-2, v1, v0) + v2)

        v0, v1, v2 = if t2 < 0.5
            F[i+1, j-1], v_right...
        else
            v_right..., F[i+1, j+2]
        end
        correction_right = 0.5 * (t2 - 0.5)^2 * (muladd(-2, v1, v0) + v2)
    end

    v0_MQS = lerp_left  + correction_left
    v1_MQS = lerp_right + correction_right

    return lerp((v0_MQS, v1_MQS), (t1,))
end
