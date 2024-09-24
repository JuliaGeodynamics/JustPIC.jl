# 2D MQS-x
@inline function MQS(F, v::NTuple{4}, t::NTuple{2}, i, j, ::Val{1})
    t1, t2 = t
    lerp_bot = lerp(v[1:2], (t1,))
    lerp_top = lerp(v[3:4], (t1,))

    v0, v1, v2 = if t1 < 0.5
        F[i - 1, j], v[1], v[2]
    else
        v[1], v[2], F[i + 2, j]
    end
    correction_bot = 0.5 * (t1 - 0.5)^2 * (muladd(-2, v1, v0) + v2)
    # correction_bot = 0.5 * (t1 - 0.5)^2 * (v0 - 2*v1 + v2)

    v0, v1, v2 = if t[1] < 0.5
        F[i - 1, j + 1], v[3], v[4]
    else
        v[3], v[4], F[i + 2, j + 1]
    end
    correction_top = 0.5 * (t1 - 0.5)^2 * (muladd(-2, v1, v0) + v2)
    # correction_top = 0.5 * (t1 - 0.5)^2 * (v0 - 2*v1 + v2)

    v0_MQS = lerp_bot + correction_bot
    v1_MQS = lerp_top + correction_top

    return lerp((v0_MQS, v1_MQS), (t2,))
end

# 2D MQS-y
@inline function MQS(F, v::NTuple{4}, t::NTuple{2}, i, j, ::Val{2})
    t1, t2 = t
    v_left = (v[1], v[3])
    v_right = (v[2], v[4])
    lerp_left = lerp(v_left, (t2,))
    lerp_right = lerp(v_right, (t2,))

    v0, v1, v2 = if t2 < 0.5
        F[i, j - 1], v_left...
    else
        v_left..., F[i, j + 2]
    end
    correction_left = 0.5 * (t2 - 0.5)^2 * (muladd(-2, v1, v0) + v2)

    v0, v1, v2 = if t2 < 0.5
        F[i + 1, j - 1], v_right...
    else
        v_right..., F[i + 1, j + 2]
    end
    correction_right = 0.5 * (t2 - 0.5)^2 * (muladd(-2, v1, v0) + v2)

    v0_MQS = lerp_left + correction_left
    v1_MQS = lerp_right + correction_right

    return lerp((v0_MQS, v1_MQS), (t1,))
end

# 3D
for i in (1, 2)
    # Val(1) -> MQS-x
    # Val(2) -> MQS-y
    @eval @inline function MQS(F, v::NTuple{8}, t::NTuple{3}, i, j, k, ::Val{$i})
        MQS_bot = MQS(F, v[1:4], t[1:2], i, j, k, Val($i))
        MQS_top = MQS(F, v[5:end], t[1:2], i, j, k, Val($i))
        return lerp((MQS_bot, MQS_top), (t[3],))
    end
end

# 3D MQS-z
@inline function MQS(F, v::NTuple{8}, t::NTuple{3}, i, j, k, ::Val{3})
    MQS_front = MQS(F, (v[1], v[2], v[5], v[6]), (t[1], t[3]), i, j, k, Val(3))
    MQS_back = MQS(F, (v[3], v[4], v[7], v[8]), (t[1], t[3]), i, j, k, Val(3))
    return lerp((MQS_front, MQS_back), (t[2],))
end

# 3D MQS-x
@inline function MQS(F, v::NTuple{4}, t::NTuple{2}, i, j, k, ::Val{1})
    t1, t2 = t
    lerp_bot = lerp(v[1:2], (t1,))
    lerp_top = lerp(v[3:4], (t1,))

    v0, v1, v2 = if t1 < 0.5
        F[i - 1, j, k], v[1], v[2]
    else
        v[1], v[2], F[i + 2, j, k]
    end
    correction_bot = 0.5 * (t1 - 0.5)^2 * (muladd(-2, v1, v0) + v2)

    v0, v1, v2 = if t[1] < 0.5
        F[i - 1, j + 1, k], v[3], v[4]
    else
        v[3], v[4], F[i + 2, j + 1, k]
    end
    correction_top = 0.5 * (t1 - 0.5)^2 * (muladd(-2, v1, v0) + v2)

    v0_MQS = lerp_bot + correction_bot
    v1_MQS = lerp_top + correction_top

    return lerp((v0_MQS, v1_MQS), (t2,))
end

# 3D MQS-y
@inline function MQS(F, v::NTuple{4}, t::NTuple{2}, i, j, k, ::Val{2})
    t1, t2 = t
    v_left = (v[1], v[3])
    v_right = (v[2], v[4])
    lerp_left = lerp(v_left, (t2,))
    lerp_right = lerp(v_right, (t2,))

    v0, v1, v2 = if t2 < 0.5
        F[i, j - 1, k], v_left...
    else
        v_left..., F[i, j + 2, k]
    end
    correction_left = 0.5 * (t2 - 0.5)^2 * (muladd(-2, v1, v0) + v2)

    v0, v1, v2 = if t2 < 0.5
        F[i + 1, j - 1, k], v_right...
    else
        v_right..., F[i + 1, j + 2, k]
    end
    correction_right = 0.5 * (t2 - 0.5)^2 * (muladd(-2, v1, v0) + v2)

    v0_MQS = lerp_left + correction_left
    v1_MQS = lerp_right + correction_right

    return lerp((v0_MQS, v1_MQS), (t1,))
end

# 3D MQS-z
@inline function MQS(F, v::NTuple{4}, t::NTuple{2}, i, j, k, ::Val{3})
    t1, t2 = t
    lerp_bot = lerp(v[1:2], (t1,))
    lerp_top = lerp(v[3:4], (t1,))

    v0, v1, v2 = if t1 < 0.5
        F[i - 1, j, k], v[1], v[2]
    else
        v[1], v[2], F[i + 2, j, k]
    end
    correction_bot = 0.5 * (t1 - 0.5)^2 * (muladd(-2, v1, v0) + v2)

    v0, v1, v2 = if t[1] < 0.5
        F[i - 1, j, k + 1], v[3], v[4]
    else
        v[3], v[4], F[i + 2, j, k + 1]
    end
    correction_top = 0.5 * (t1 - 0.5)^2 * (muladd(-2, v1, v0) + v2)

    v0_MQS = lerp_bot + correction_bot * 1
    v1_MQS = lerp_top + correction_top * 1

    return lerp((v0_MQS, v1_MQS), (t2,))
end
