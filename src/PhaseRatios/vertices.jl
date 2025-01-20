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
        ratio_vertices, pxi::NTuple{3}, xvi::NTuple{3}, di::NTuple{3, T}, phases
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
            # check if it's within half cell
            prod(x -> abs(x[1] - x[2]) ≥ x[3] / 2, zip(p, cell_vertex, di)) && continue

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
        ratio_vertices, pxi::NTuple{2}, xvi::NTuple{2}, di::NTuple{2, T}, phases
    ) where {T}

    # index corresponding to the cell center
    cell_vertex = xvi[1][I[1]], xvi[2][I[2]]
    ni = size(phases)
    NC = nphases(ratio_vertices)
    w = ntuple(_ -> zero(T), NC)

    for offsetᵢ in -1:0, offsetⱼ in -1:0
        i_cell = I[1] + offsetᵢ
        !(0 < i_cell < ni[1] + 1) && continue
        j_cell = I[2] + offsetⱼ
        !(0 < j_cell < ni[2] + 1) && continue

        cell_index = i_cell, j_cell

        for ip in cellaxes(phases)
            p = @index(pxi[1][ip, cell_index...]), @index(pxi[2][ip, cell_index...])
            any(isnan, p) && continue
            # check if it's within half cell
            prod(x -> abs(x[1] - x[2]) ≤ x[3] / 2, zip(p, cell_vertex, di)) && continue
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
