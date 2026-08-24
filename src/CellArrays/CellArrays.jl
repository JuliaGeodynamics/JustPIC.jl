"""
    cellnum(A::CellArray)

Return the number of storage slots inside each logical cell of `A`.

For particle containers this is the number of particle slots reserved per grid
cell, including inactive slots.
"""
@inline cellnum(A::CellArray) = prod(cellsize(A))

"""
    cellaxes(A)

Return the one-based axes used to iterate over the entries inside each
`CellArray` cell.

This is the preferred helper for loops over particle slots because it works for
both scalar and multi-entry cell storage.
"""
@inline cellaxes(A) = map(Base.oneto, cellnum(A))

"""
    new_empty_cell(A::CellArray)

Create a zero-valued cell payload with the same element type as `A`.
"""
@inline new_empty_cell(::CellArray{T, N}) where {T, N} = zeros(T)
