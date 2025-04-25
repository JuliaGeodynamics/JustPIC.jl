function update_cell_halo!(
        x::Vararg{CellArray{S, N, D, A}, NA}
    ) where {NA, S, N, D, A <: AbstractArray}
    ni = size(x[1])
    tmp = @fill(0e0, ni..., eltype = eltype(x[1].data))

    for xᵢ in x
        for ip in cellaxes(xᵢ)
            @parallel (@idx ni) copy_field!(tmp, xᵢ, ip)
            update_halo!(tmp)
            @parallel (@idx ni) copy_field!(xᵢ, tmp, ip)
        end
    end
    return nothing
end

@parallel_indices (I...) function copy_field!(A::CellArray, B::AbstractArray, ip)
    @inbounds @index A[ip, I...] = B[I...]
    return nothing
end

@parallel_indices (I...) function copy_field!(B::AbstractArray, A::CellArray, ip)
    @inbounds B[I...] = @index A[ip, I...]
    return nothing
end
