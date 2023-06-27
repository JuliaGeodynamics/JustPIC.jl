## ND LINEAR INTERPOLATION KERNELS

"""
    lerp(t, v0, v1)

Linear interpolation between `f(x0)=v0` and `f(x1)=v1` at the normalized coordinate t
"""
@inline lerp(t, v0, v1) = fma(t, v1, fma(-t, v0, v0))

"""
    bilinear(tx, ty, v00, v10, v01, v11)

Linear interpolation between `f(x0,y0)=v0` and `f(x1,y1)=v1` at the normalized coordinates `tx`, `ty`
"""
@inline function ndlinear(tx, ty, v00, v10, v01, v11)
    return lerp(ty, lerp(tx, v00, v10), lerp(tx, v01, v11))
end

@inline function ndlinear(t::NTuple{2,T1}, v::NTuple{4,T2}) where {T1,T2}
    return ndlinear(t[1], t[2], v[1], v[2], v[3], v[4])
end

bilinear(tx, ty, v00, v10, v01, v11) = ndlinear(tx, ty, v00, v10, v01, v11)

"""
    trilinear(tx, ty, tz, v000, v100, v001, v101, v010, v110, v011, v111) 

Linear interpolation between `f(x0,y0,z0)=v0` and `f(x1,y1,z1)=v1` at the normalized coordinates `tx`, `ty`, `tz`
"""
@inline function ndlinear(tx, ty, tz, v000, v100, v001, v101, v010, v110, v011, v111)
    return lerp(
        ty,
        ndlinear(tx, tz, v000, v100, v001, v101),
        ndlinear(tx, tz, v010, v110, v011, v111),
    )
end

@inline function ndlinear(t::NTuple{3,T1}, v::NTuple{8,T2}) where {T1,T2}
    return ndlinear(t[1], t[2], t[3], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8])
end

@inline function trilinear(tx, ty, tz, v000, v100, v001, v101, v010, v110, v011, v111)
    return ndlinear(tx, ty, tz, v000, v100, v001, v101, v010, v110, v011, v111)
end
