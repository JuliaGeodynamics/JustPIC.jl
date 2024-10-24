
function phase_ratios_midpoint!(phase_midpoint, particles, xci::NTuple{N}, phases, dimension) where N
    ni = size(phases)
    di = compute_dx(xci)

    offsets = if N == 2
        offsets = if dimension === :x
            (1, 0)
        elseif dimension === :y
            (0, 1)
        else
            throw("Unknown dimensions. Valid dimensions are :x, :y")
        end
    elseif N == 3
        offsets = if dimension === :x
            (1, 0, 0)
        elseif dimension === :y
            (0, 1, 0)
        elseif dimension === :z
            (0, 0, 1)
        else
            throw("Unknown dimensions. Valid dimensions are :x, :y, :z")
        end
    end

    @parallel (@idx ni) phase_ratios_midpoint_kernel!(
        phase_midpoint, particles.coords, xci, di, phases, offsets
    )
    return nothing
end

@parallel_indices (I...) function phase_ratios_midpoint_kernel!(
    ratio_midpoints, pxi::NTuple{N}, xci::NTuple{N}, di::NTuple{N,T}, phases, offsets
) where {N,T}

    # index corresponding to the cell center
    cell_center   = getindex.(xci, I)
    cell_midpoint = @. cell_center + di * offsets / 2
    ni = size(phases)
    NC = nphases(ratio_midpoints)
    w = ntuple(_ -> zero(T), NC)

    # general case
    for offsetsᵢ in zip(ntuple(_->0, Val(N)), (offsets))
        cell_index = min.(I .+ offsetsᵢ, ni)
        all(@. 0 < cell_index < ni + 1) || continue

        for ip in cellaxes(phases)
            p = ntuple(Val(N)) do i
                @index pxi[i][ip, cell_index...]
            end
            any(isnan, p) && continue
            # check if it's within half cell
            prod(x -> abs(x[1] - x[2]) ≤ x[3] / 2, zip(p, cell_midpoint, di)) && continue
            x = @inline bilinear_weight(cell_midpoint, p, di)
            ph_local = @index phases[ip, cell_index...]
            # this is doing sum(w * δij(i, phase)), where δij is the Kronecker delta
            w = w .+ x .* ntuple(j -> (ph_local == j), NC)
        end
    end

    w = w .* inv(sum(w))
    for ip in cellaxes(ratio_midpoints)
        @index ratio_midpoints[ip, (I.+offsets)...] = w[ip]
    end

    # handle i == 1 or j == 1
    boundary_offset = @.  1 * offsets * I
    isboundary = any(isone, boundary_offset)
    if isboundary
        # index corresponding to the cell center
        cell_midpoint = @. cell_center - di * offsets / 2
        w = ntuple(_ -> zero(T), NC)

        for ip in cellaxes(phases)
            p = ntuple(Val(N)) do i
                @index pxi[i][ip, I...]
            end
            any(isnan, p) && continue
            # check if it's within half cell
            prod(x -> abs(x[1] - x[2]) ≤ x[3] / 2, zip(p, cell_midpoint, di)) && continue
            x = @inline bilinear_weight(cell_midpoint, p, di)
            ph_local = @index phases[ip, I...]
            # this is doing sum(w * δij(i, phase)), where δij is the Kronecker delta
            w = w .+ x .* ntuple(j -> (ph_local == j), NC)
        end
        w = w .* inv(sum(w))
        for ip in cellaxes(ratio_midpoints)
            @index ratio_midpoints[ip, I...] = w[ip]
        end
    end

    return nothing
end