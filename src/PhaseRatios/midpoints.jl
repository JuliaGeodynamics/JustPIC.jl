function phase_ratios_midpoint!(phase_midpoint, particles, xci::NTuple{N}, phases, dimension) where N
    ni      = size(phases)
    di      = compute_dx(xci)
    offsets = midpoint_offset(Val(N), dimension)

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
    ni            = size(phases)
    NC            = nphases(ratio_midpoints)
    w             = ntuple(_ -> zero(T), NC)

    # general case
    for offsetsᵢ in (ntuple(_->0, Val(N)), offsets)
        cell_index = min.(I .+ offsetsᵢ, ni)
        all(@. 0 < cell_index < ni + 1) || continue

        for ip in cellaxes(phases)
            p = get_particle_coords(pxi, ip, cell_index...)
            any(isnan, p) && continue
            # check if it's within half cell
            isinhalfcell(p, cell_midpoint, di) && continue
            x        = @inline bilinear_weight(cell_midpoint, p, di)
            ph_local = @index phases[ip, cell_index...]
            # this is doing sum(w * δij(i, phase)), where δij is the Kronecker delta
            w        = accumulate_weight(w, x, ph_local, Val(NC))
        end
    end

    w = w .* inv(sum(w))
    for ip in cellaxes(ratio_midpoints)
        @index ratio_midpoints[ip, (I.+offsets)...] = w[ip]
    end
    
    if isboundary(offsets, I)
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
            x        = @inline bilinear_weight(cell_midpoint, p, di)
            ph_local = @index phases[ip, I...]
            # this is doing sum(w * δij(i, phase)), where δij is the Kronecker delta
            w        = accumulate_weight(w, x, ph_local, Val(NC))
        end
        w = w .* inv(sum(w))
        for ip in cellaxes(ratio_midpoints)
            @index ratio_midpoints[ip, I...] = w[ip]
        end
    end

    return nothing
end

@inline accumulate_weight(w, x, phase, ::Val{N}) where N = w .+ x .* ntuple(j -> (phase == j), Val(N))

@inline isinhalfcell(p, cell_midpoint, di) = prod(x -> abs(x[1] - x[2]) ≤ x[3] / 2, zip(p, cell_midpoint, di))

@inline function midpoint_offset(::Val{3}, s::Symbol)
    offsets = if dimension === :x
        (1, 0)
    elseif dimension === :y
        (0, 1)
    else
        throw("Unknown dimensions. Valid dimensions are :x, :y")
    end
end

@inline function midpoint_offset(::Val[3], s::Symbol)
    offsets = if dimension === :x
        (1, 0, 0)
    elseif dimension === :y
        (0, 1, 0)
    elseif dimension === :z
        (0, 0, 1)
    elseif dimension === :xy
        (1, 1, 0)
    elseif dimension === :yz
        (0, 1, 1)
    elseif dimension === :xz
        (1, 0, 1)
    else
        throw("Unknown dimensions. Valid dimensions are :x, :y, :z, :xy, :yz, :xz")
    end
end

@generated function isboundary(offsets::NTuple{N}, I::NTuple{N}) where {N}
    quote
        @inline 
        Base.@nany $N i -> @inbounds isone(offsets[i] * I[i])
    end
end
