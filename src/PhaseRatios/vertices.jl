## Kernels to compute phase ratios at the vertices


function phase_ratios_vertex!(phase_ratios::JustPIC.PhaseRatios, particles, phases)
    ni = inner_size(phase_ratios.vertex)

    launch!(
        ka_backend(phase_ratios), phase_ratios_vertex_kernel!, ni,
        phase_ratios.vertex, particles.coords, particles.xvi, particles.di.vertex, phases
    )
    return nothing
end

@kernel function phase_ratios_vertex_kernel!(
        ratio_vertices, pxi::NTuple{3}, xvi::NTuple{3, T}, dᵢ, phases
    ) where {T}
    I = @index(Global, NTuple)
    I_inner = I .+ 1

    # index corresponding to the cell center
    cell_vertex = xvi[1][I_inner[1]], xvi[2][I_inner[2]], xvi[3][I_inner[3]]
    ni = size(phases)
    NC = nphases(ratio_vertices)
    w = ntuple(_ -> zero(eltype(eltype(ratio_vertices))), NC)

    for offsetᵢ in -1:0, offsetⱼ in -1:0, offsetₖ in -1:0
        i_cell = I_inner[1] + offsetᵢ
        0 < i_cell < ni[1] + 1 || continue
        j_cell = I_inner[2] + offsetⱼ
        0 < j_cell < ni[2] + 1 || continue
        k_cell = I_inner[3] + offsetₖ
        0 < k_cell < ni[3] + 1 || continue

        cell_index = i_cell, j_cell, k_cell
        di = @dxi dᵢ cell_index...

        for ip in cellaxes(phases)
            p = CAI.@index(pxi[1][ip, cell_index...]),
                CAI.@index(pxi[2][ip, cell_index...]),
                CAI.@index(pxi[3][ip, cell_index...])
            any(isnan, p) && continue
            # check if it's within half cell
            tmp = false
            for i in eachindex(p)
                if abs(p[i] - cell_vertex[i]) ≥ di[i] / 2
                    tmp = true
                    break
                end
            end
            tmp && continue
            x = @inline bilinear_weight(cell_vertex, p, di)
            ph_local = CAI.@index phases[ip, cell_index...]
            # this is doing sum(w * δij(i, phase)), where δij is the Kronecker delta
            w = w .+ x .* ntuple(j -> (ph_local == j), NC)
        end
    end

    w = w .* inv(sum(w))
    for ip in cellaxes(ratio_vertices)
        CAI.@index ratio_vertices[ip, I_inner...] = w[ip]
    end
end

@kernel function phase_ratios_vertex_kernel!(
        ratio_vertices, pxi::NTuple{2}, xvi::NTuple{2}, dᵢ, phases
    )
    I = @index(Global, NTuple)
    I_inner = I .+ 1

    # index corresponding to the cell center
    cell_vertex = xvi[1][I_inner[1]], xvi[2][I_inner[2]]
    ni = size(phases)
    NC = nphases(ratio_vertices)
    w = ntuple(_ -> zero(eltype(eltype(ratio_vertices))), NC)

    for offsetᵢ in -1:0, offsetⱼ in -1:0
        i_cell = I_inner[1] + offsetᵢ
        !(0 < i_cell < ni[1] + 1) && continue
        j_cell = I_inner[2] + offsetⱼ
        !(0 < j_cell < ni[2] + 1) && continue

        cell_index = i_cell, j_cell
        di = @dxi dᵢ cell_index...

        for ip in cellaxes(phases)
            p = CAI.@index(pxi[1][ip, cell_index...]), CAI.@index(pxi[2][ip, cell_index...])
            any(isnan, p) && continue
            # check if it's within half cell
            tmp = false
            for i in eachindex(p)
                if abs(p[i] - cell_vertex[i]) ≥ di[i] / 2
                    tmp = true
                    break
                end
            end
            tmp && continue

            x = @inline bilinear_weight(cell_vertex, p, di)
            ph_local = CAI.@index phases[ip, cell_index...]
            # this is doing sum(w * δij(i, phase)), where δij is the Kronecker delta
            w = w .+ x .* ntuple(j -> (ph_local == j), NC)
        end
    end

    w = w .* inv(sum(w))
    for ip in cellaxes(ratio_vertices)
        CAI.@index ratio_vertices[ip, I_inner...] = w[ip]
    end
end
