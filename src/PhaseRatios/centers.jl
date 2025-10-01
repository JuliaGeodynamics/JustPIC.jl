## Kernels to compute phase ratios at the centers

function phase_ratios_center!(phase_ratios::JustPIC.PhaseRatios, particles, xci, phases)
    ni = size(phases)
    di = compute_dx(xci)

    @parallel (@idx ni) phase_ratios_center_kernel!(
        phase_ratios.center, particles.coords, xci, di, phases
    )
    return nothing
end

@parallel_indices (I...) function phase_ratios_center_kernel!(
        ratio_centers, pxi::NTuple{N}, xci::NTuple{N}, di::NTuple{N}, phases
    ) where {N}

    # index corresponding to the cell center
    cell_center = ntuple(i -> xci[i][I[i]], Val(N))
    # phase ratios weights (∑w = 1.0)
    w = phase_ratio_weights(
        getindex.(pxi, I...), phases[I...], cell_center, @dxi(di, I...), nphases(ratio_centers)
    )
    # update phase ratios array
    for k in 1:numphases(ratio_centers)
        @index ratio_centers[k, I...] = w[k]
    end

    return nothing
end
