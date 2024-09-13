# ## Fallbacks
# import Base: getindex, setindex!

# @inline element(A::CuArray, I::Vararg{Int,N}) where {N} = getindex(A, I...)
# @inline function setelement!(A::CuArray, x::Number, I::Vararg{Int,N}) where {N}
#     return setindex!(A, x, I...)
# end
