
@inline cellnum(A::CellArray) = prod(cellsize(A))
@inline cellaxes(A) = map(Base.oneto, cellnum(A))
@inline new_empty_cell(::CellArray{T,N}) where {T,N} = zeros(T)
