struct MarkerChain{N,I,T1,T2} <: AsbtractParticles
    coords::NTuple{N,T1}
    index::T2
    nxcell::I
    max_xcell::I
    min_xcell::I

    function MarkerChain(
        coords::NTuple{N,T1}, index, nxcell::I, max_xcell::I, min_xcell::I
    ) where {N,I,T1}

        # types
        T2 = typeof(index)
        return new{N,I,T1,T2}(coords, index, nxcell, max_xcell, min_xcell)
    end
end

# Sorting 

function sort!(particles::MarkerChain)
    sort!(particles.index)
    return nothing
end

@inline skip(index <: CellArray, ip, I::Vararg{Int64,N}) where {N} =
    @inbounds !@cell(index[ip, I...])

@parallel_indices (I...) function sort!(coords::NTuple{N,T}, index) where {N,T}
    for ip in cellaxes(coords[1])
        skip(index, ip, I...) && continue
    end

    return nothing
end

# Sorts A and permutes B accordingly
function insert_sort!(A <: CellArray, B <: CellArray, I::Vararg{Int64,N}) where {N}
    i = 2
    n = cellnum(A)
    @inbounds while i ≤ n
        x, y = @cell(A[i, I...]), @cell(B[i, I...])
        j = i - 1
        while j > 1 && A[j] > x
            # manually do the swap
            a, a1 = @cell(A[j, I...]), @cell(A[j + 1, I...])
            b, b1 = @cell(B[j, I...]), @cell(B[j + 1, I...])
            @cell A[j + 1, I...] = a
            @cell A[j, I...] = a1
            @cell B[j + 1, I...] = b
            @cell B[j, I...] = b1
            j -= 1
        end
        @cell A[j + 1, I...] = x
        @cell B[j + 1, I...] = y
        i += 1
    end

    i = 2
    @inbounds while i ≤ n && A[i - 1] ≥ A[i]
        # manually do the swap
        a, a1 = @cell(A[j, I...]), @cell(A[j - 1, I...])
        b, b1 = @cell(B[j, I...]), @cell(B[j - 1, I...])
        @cell A[j - 1, I...] = a
        @cell A[j, I...] = a1
        @cell B[j - 1, I...] = b
        @cell B[j, I...] = b1
        i += 1
    end

    return nothing
end
