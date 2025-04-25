@generated function update_cell_halo!(
        x::Vararg{CellArray{S, N, D, A}, NA}
    ) where {NA, S, N, D, A <: AbstractArray}
    quote 
        ni = size(x[1])
        tmp = @fill(0, ni..., eltype = eltype(x[1].data))

        Base.@nexprs $N i-> begin
            xᵢ = x[i]
            for ip in cellaxes(xᵢ)
                copyto!(tmp, field(xᵢ, ip))
                update_halo!(tmp)
                @parallel (@idx ni) copy_field!(xᵢ, tmp, ip)
            end
        end
        return nothing
    end
end

@parallel_indices (I...) function copy_field!(A::CellArray, B::AbstractArray, ip)
    @inbounds @index A[ip, I...] = B[I...]
    return nothing
end
