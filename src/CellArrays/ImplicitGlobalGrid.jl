function update_cell_halo!(x::Vararg{CellArray,N}) where {N}
    ni = size(x[1])
    tmp = @fill(0, ni..., eltype = eltype(x[1].data))

    for i in 1:N
        for ip in cellaxes(x[i])
            tmp .= field(x[i], ip)
            update_halo!(tmp)
            @parallel (@idx ni) copy_field!(x[i], tmp, ip)
        end
    end
    return nothing
end

@parallel_indices (I...) function copy_field!(
    A::CellArray, B::AbstractArray{T,2}, ip
) where {T}
    @cell A[ip, I...] = B[I...]
    return nothing
end
