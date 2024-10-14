function update_phase_ratios!(phase_ratios::JustPIC.PhaseRatios, particles, xci, xvi, phases)
    phase_ratios_center!(phase_ratios, particles, xci, phases)
    phase_ratios_vertex!(phase_ratios, particles, xvi, phases)
    return nothing
end

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
    ratio_centers, pxi::NTuple{N,T1}, xci::NTuple{N,T2}, di::NTuple{N,T3}, phases
) where {N,T1,T2,T3}

    # index corresponding to the cell center
    cell_center = ntuple(i -> xci[i][I[i]], Val(N))
    # phase ratios weights (∑w = 1.0)
    w = phase_ratio_weights(
        getindex.(pxi, I...), phases[I...], cell_center, di, nphases(ratio_centers)
    )
    # update phase ratios array
    for k in 1:numphases(ratio_centers)
        @index ratio_centers[k, I...] = w[k]
    end

    return nothing
end

## Kernels to compute phase ratios at the vertices

function phase_ratios_vertex!(phase_ratios::JustPIC.PhaseRatios, particles, xvi, phases)
    ni = size(phases) .+ 1
    di = compute_dx(xvi)

    @parallel (@idx ni) phase_ratios_vertex_kernel!(
        phase_ratios.vertex, particles.coords, xvi, di, phases
    )
    return nothing
end

@parallel_indices (I...) function phase_ratios_vertex_kernel!(
    ratio_vertices, pxi::NTuple{3}, xvi::NTuple{3}, di::NTuple{3,T}, phases
) where {T}

    # index corresponding to the cell center
    cell_vertex = xvi[1][I[1]], xvi[2][I[2]], xvi[3][I[3]]
    ni = size(phases)
    NC = nphases(ratio_vertices)
    w = ntuple(_ -> zero(T), NC)

    for offsetᵢ in -1:0, offsetⱼ in -1:0, offsetₖ in -1:0
        i_cell = I[1] + offsetᵢ
        0 < i_cell < ni[1] + 1 || continue
        j_cell = I[2] + offsetⱼ
        0 < j_cell < ni[2] + 1 || continue
        k_cell = I[3] + offsetₖ
        0 < k_cell < ni[3] + 1 || continue

        cell_index = i_cell, j_cell, k_cell

        for ip in cellaxes(phases)
            p = @index(pxi[1][ip, cell_index...]),
            @index(pxi[2][ip, cell_index...]),
            @index(pxi[3][ip, cell_index...])
            any(isnan, p) && continue
            x = @inline bilinear_weight(cell_vertex, p, di)
            ph_local = @index phases[ip, cell_index...]
            # this is doing sum(w * δij(i, phase)), where δij is the Kronecker delta
            w = w .+ x .* ntuple(j -> (ph_local == j), NC)
        end
    end

    w = w .* inv(sum(w))
    for ip in cellaxes(ratio_vertices)
        @index ratio_vertices[ip, I...] = w[ip]
    end

    return nothing
end

@parallel_indices (I...) function phase_ratios_vertex_kernel!(
    ratio_vertices, pxi::NTuple{2}, xvi::NTuple{2}, di::NTuple{2,T}, phases
) where {T}

    # index corresponding to the cell center
    cell_vertex = xvi[1][I[1]], xvi[2][I[2]]
    ni = size(phases)
    NC = nphases(ratio_vertices)
    w = ntuple(_ -> zero(T), NC)

    for offsetᵢ in -1:0, offsetⱼ in -1:0
        i_cell = I[1] + offsetᵢ
        0 < i_cell < ni[1] + 1 || continue
        j_cell = I[2] + offsetⱼ
        0 < j_cell < ni[2] + 1 || continue

        cell_index = i_cell, j_cell

        for ip in cellaxes(phases)
            p = @index(pxi[1][ip, cell_index...]), @index(pxi[2][ip, cell_index...])
            any(isnan, p) && continue
            x = @inline bilinear_weight(cell_vertex, p, di)
            ph_local = @index phases[ip, cell_index...]
            # this is doing sum(w * δij(i, phase)), where δij is the Kronecker delta
            w = ntuple(j -> (ph_local == j) * x[i] + w[i], NC)
        end
    end

    w = w .* inv(sum(w))
    for ip in cellaxes(ratio_vertices)
        @index ratio_vertices[ip, I...] = w[ip]
    end

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
        w = ntuple(j -> (ph_local == j) * x[i] + w[i], Val(NC))
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
