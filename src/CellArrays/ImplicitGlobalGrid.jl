@generated function update_cell_halo!(
        x::Vararg{CellArray{S, N, D, A}, NA}
    ) where {NA, S, N, D, A <: AbstractArray}
    ni = size(x[1])
    tmp = @fill(0.0e0, ni..., eltype = eltype(x[1].data))

    for xᵢ in x
        for ip in cellaxes(xᵢ)
            @parallel (@idx ni) move_CellArray_to_Array!(tmp, xᵢ, ip)
            update_halo!(tmp)
            @parallel (@idx ni) move_Array_to_CellArray!(xᵢ, tmp, ip)
        end
        return nothing
    end
end

@parallel_indices (I...) function move_Array_to_CellArray!(A::CellArray, B::AbstractArray, ip)
    @inbounds @index A[ip, I...] = B[I...]
    return nothing
end

@parallel_indices (I...) function move_CellArray_to_Array!(B::AbstractArray, A::CellArray, ip)
    @inbounds B[I...] = @index A[ip, I...]
    return nothing
end
