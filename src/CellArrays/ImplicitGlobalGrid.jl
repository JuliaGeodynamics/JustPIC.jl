"""
    update_cell_halo!(x)

Update the halo of the `CellArray` `x`
"""
function update_cell_halo!(x)
    ni = size(x)
    tmp = @fill(0, ni..., eltype=eltype(x.data))

    for ip in cellaxes(x)
        tmp .= field(x, ip)
        update_halo!(tmp)
        @parallel (@range ni) copy_field!(x, tmp, ip)
    end

end

@parallel_indices (i, j) function copy_field!(A::CellArray, B::AbstractArray{T, 2}, ip) where T
    @cell A[ip, i, j] = B[i, j]
    return nothing
end

@parallel_indices (i, j, k) function copy_field!(A::CellArray, B::AbstractArray{T, 3}, ip) where T
    @cell A[ip, i, j, k] = B[i, j, k]
    return nothing
end
