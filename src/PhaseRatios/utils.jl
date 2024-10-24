function update_phase_ratios!(phase_ratios::JustPIC.PhaseRatios, particles, xci, xvi, phases)
    phase_ratios_center!(phase_ratios, particles, xci, phases)
    phase_ratios_vertex!(phase_ratios, particles, xvi, phases)
    return nothing
end

## interpolation kernels

function phase_ratio_weights(
    pxi::NTuple{NP,C}, ph::SVector{N1,T}, cell_center, di, ::Val{NC}
) where {N1,NC,NP,T,C}

    # Initiaze phase ratio weights (note: can't use ntuple() here because of the @generated function)
    w = ntuple(_ -> zero(T), Val(NC))
    # sumw = zero(T)

    for i in eachindex(ph)
        p = getindex.(pxi, i)
        isnan(first(p)) && continue
        x = @inline bilinear_weight(cell_center, p, di)
        # sumw += x # reduce
        ph_local = ph[i]
        # this is doing sum(w * δij(i, phase)), where δij is the Kronecker delta
        w = w .+ x .* ntuple(j -> (ph_local == j), Val(NC))
    end
    w = w .* inv(sum(w))
    return w
end

@generated function bilinear_weight(
    a::NTuple{N,T}, b::NTuple{N,T}, di::NTuple{N,T}
) where {N,T}
    quote
        Base.@_inline_meta
        val = one($T)
        Base.Cartesian.@nexprs $N i ->
            @inbounds val *= muladd(-abs(a[i] - b[i]), inv(di[i]), one($T))
        return val
    end
end
