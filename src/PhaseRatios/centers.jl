## Kernels to compute phase ratios at the centers

function phase_ratios_center!(phase_ratios::JustPIC.PhaseRatios, particles, phases)
    ni = size(phase_ratios.center)

    @parallel (@idx ni) phase_ratios_center_kernel!(
        phase_ratios.center, particles.coords, particles.xci, particles.di.vertex, phases
    )
    return nothing
end

@parallel_indices (I...) function phase_ratios_center_kernel!(
        ratio_centers, pxi::NTuple{N}, xci::NTuple{N}, dᵢ::NTuple{N}, phases
    ) where {N}
    I_inner = I .+ 1

    # compute dxi
    di = @dxi dᵢ I_inner...
    # index corresponding to the cell center
    cell_center = ntuple(i -> xci[i][I_inner[i]], Val(N))
    # phase ratios weights (∑w = 1.0)
    w = phase_ratio_weights(
        getindex.(pxi, I_inner...), phases[I_inner...], cell_center, di, nphases(ratio_centers)
    )
    # update phase ratios array
    for k in 1:numphases(ratio_centers)
        @index ratio_centers[k, I_inner...] = w[k]
    end

    return nothing
end
