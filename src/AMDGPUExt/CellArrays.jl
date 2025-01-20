## Fallbacks
import Base: getindex, setindex!

@inline element(A::ROCArray, I::Vararg{Int,N}) where {N} = getindex(A, I...)
@inline function setelement!(A::ROCArray, x::Number, I::Vararg{Int,N}) where {N}
    return setindex!(A, x, I...)
end
