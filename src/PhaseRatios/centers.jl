## Kernels to compute phase ratios at the centers

function phase_ratios_center!(phase_ratios::JustPIC.PhaseRatios, particles, phases)
    ni = size(phases)

    launch!(
        ka_backend(phase_ratios), phase_ratios_center_kernel!, ni,
        phase_ratios.center, particles.coords, particles.xci, particles.di.vertex, phases
    )
    return nothing
end

@kernel function phase_ratios_center_kernel!(
        ratio_centers, pxi::NTuple{N}, xci::NTuple{N}, dᵢ::NTuple{N}, phases
    ) where {N}
    I = @index(Global, NTuple)

    # compute dxi
    di = @dxi dᵢ I...
    # index corresponding to the cell center
    cell_center = ntuple(i -> xci[i][I[i]], Val(N))
    # phase ratios weights (∑w = 1.0)
    w = phase_ratio_weights(
        getindex.(pxi, I...), phases[I...], cell_center, di, nphases(ratio_centers)
    )
    # update phase ratios array
    for k in 1:numphases(ratio_centers)
        CAI.@index ratio_centers[k, I...] = w[k]
    end
end
