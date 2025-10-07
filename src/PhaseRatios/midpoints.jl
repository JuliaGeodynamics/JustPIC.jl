## CELL FACES: AKA VELOCITY-NODES

phase_ratios_face!(phase_face, particles, xci::NTuple{N}, phases, dimension) where {N} = phase_ratios_face!(phase_face, particles, xci::NTuple{N}, phases, dimension, compute_dx(xci))

function phase_ratios_face!(
        phase_face, particles, xci::NTuple{N}, phases, dimension, di
    ) where {N}
    ni = size(phases)
    offsets = face_offset(Val(N), dimension)

    @parallel (@idx ni) phase_ratios_face_kernel!(
        phase_face, particles.coords, xci, di, phases, offsets
    )
    return nothing
end

@parallel_indices (I...) function phase_ratios_face_kernel!(
        ratio_faces, pxi::NTuple{N}, xci::NTuple{N}, dxi::NTuple{N, T}, phases, offsets
    ) where {N, T}

    di = @dxi(dxi, I...)
    # index corresponding to the cell center
    cell_center = getindex.(xci, I)
    cell_face = @. cell_center + di * offsets / 2
    ni = size(phases)
    NC = nphases(ratio_faces)
    w = ntuple(_ -> zero(T), NC)

    # general case
    for offsetsᵢ in (ntuple(_ -> 0, Val(N)), offsets)
        cell_index = min.(I .+ offsetsᵢ, ni)
        all(@. 0 < cell_index < ni + 1) || continue

        for ip in cellaxes(phases)
            p = get_particle_coords(pxi, ip, cell_index...)
            any(isnan, p) && continue
            # check if it's within half cell
            isinhalfcell(p, cell_face, di) || continue
            x = @inline bilinear_weight(cell_face, p, di)
            ph_local = @index phases[ip, cell_index...]
            # this is doing sum(w * δij(i, phase)), where δij is the Kronecker delta
            w = accumulate_weight(w, x, ph_local, NC)
        end
    end

    w = w .* inv(sum(w))
    for ip in cellaxes(ratio_faces)
        @index ratio_faces[ip, (I .+ offsets)...] = w[ip] * !isnan(w[ip]) # make it zero if there are NaNs (means no particles within velocity half cell)
    end

    if isboundary(offsets, I)
        # index corresponding to the cell center
        cell_face = @. cell_center - di * offsets / 2
        w = ntuple(_ -> zero(T), NC)

        for ip in cellaxes(phases)
            p = get_particle_coords(pxi, ip, I...)
            any(isnan, p) && continue
            # check if it's within half cell
            isinhalfcell(p, cell_face, di) || continue
            x = @inline bilinear_weight(cell_face, p, di)
            ph_local = @index phases[ip, I...]
            # this is doing sum(w * δij(i, phase)), where δij is the Kronecker delta
            w = accumulate_weight(w, x, ph_local, NC)
        end
        w = w .* inv(sum(w))
        for ip in cellaxes(ratio_faces)
            @index ratio_faces[ip, I...] = w[ip] * !isnan(w[ip]) # make it zero if there are NaNs (means no particles within velocity half cell)
        end
    end

    return nothing
end

@inline function face_offset(::Val{2}, dimension::Symbol)
    return offsets = if dimension === :x
        (1, 0)
    elseif dimension === :y
        (0, 1)
    else
        throw("Unknown dimensions. Valid dimensions are :x, :y")
    end
end

@inline function face_offset(::Val{3}, dimension::Symbol)
    return offsets = if dimension === :x
        (1, 0, 0)
    elseif dimension === :y
        (0, 1, 0)
    elseif dimension === :z
        (0, 0, 1)
    else
        throw("Unknown dimensions. Valid dimensions are :x, :y, :z, :xy, :yz, :xz")
    end
end

## MIDPOINTS: AKA SHEAR STRESS-NODES (ONLY IN 3D)

function phase_ratios_midpoint!(
        phase_midpoint, particles, xci::NTuple{N}, phases, dimension
    ) where {N}
    ni = size(phases)
    di = compute_dx(xci)
    offsets = midpoint_offset(Val(N), dimension)

    @parallel (@idx ni) phase_ratios_midpoint_kernel!(
        phase_midpoint, particles.coords, xci, di, phases, offsets
    )
    return nothing
end

@parallel_indices (I...) function phase_ratios_midpoint_kernel!(
        ratio_midpoints, pxi::NTuple, xci::NTuple, di::NTuple, phases, offsets
    )
    _phase_ratios_midpoint_kernel!(ratio_midpoints, pxi, xci, @dxi(di, I...), phases, offsets, I...)
    return nothing
end

function _phase_ratios_midpoint_kernel!(
        ratio_midpoints,
        pxi::NTuple{N},
        xci::NTuple{N},
        di::NTuple{N, T},
        phases,
        offsets,
        I::Vararg{Int, N},
    ) where {N, T}
    MASK_3D = (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)

    # index corresponding to the cell center
    cell_center = getindex.(xci, I)
    cell_midpoint = @. cell_center + di * offsets / 2
    ni = size(phases)
    nm = size(ratio_midpoints)
    NC = nphases(ratio_midpoints)
    w = ntuple(_ -> zero(T), NC)

    # general case
    for mask in MASK_3D
        offsetsᵢ = offsets .* mask
        cell_index = min.(I .+ offsetsᵢ, ni)
        all(cell_index .≤ nm) || continue

        for ip in cellaxes(phases)
            p = get_particle_coords(pxi, ip, cell_index...)
            any(isnan, p) && continue
            # check if it's within half cell
            isinhalfcell(p, cell_midpoint, di) || continue
            x = @inline bilinear_weight(cell_midpoint, p, di)
            ph_local = @index phases[ip, cell_index...]
            # this is doing sum(w * δij(i, phase)), where δij is the Kronecker delta
            w = accumulate_weight(w, x, ph_local, NC)
        end
    end

    w = w .* inv(sum(w))
    for ip in cellaxes(ratio_midpoints)
        @index ratio_midpoints[ip, (I .+ offsets)...] = w[ip] * !isnan(w[ip]) # make it zero if there are NaNs (means no particles within half cells)
    end

    if isboundary(offsets, I)
        offset_boundary = lastboundary_offset(offsets, I, ni)
        for offset_boundaryᵢ in ((0, 0, 0), offset_boundary)
            all(x -> x === false, offset_boundaryᵢ) && continue # skip if not last boundary

            midpoint_index = I .+ offset_boundaryᵢ
            # index corresponding to the cell center
            flip_sign_mask = (0, 0, 0) .- offset_boundary # need to add dxi if we are in the last boundary
            cell_midpoint = @. cell_center - (di * offsets * flip_sign_mask) / 2
            w = ntuple(_ -> zero(T), NC)

            for mask in MASK_3D
                offsetsᵢ = offsets .* mask
                cell_index = min.(I .+ offsetsᵢ, ni)
                all(cell_index .≤ nm) || continue
                for ip in cellaxes(phases)
                    p = get_particle_coords(pxi, ip, cell_index...)
                    any(isnan, p) && continue
                    # check if it's within half cell
                    isinhalfcell(p, cell_midpoint, di) || continue
                    x = @inline bilinear_weight(cell_midpoint, p, di)
                    ph_local = @index phases[ip, cell_index...]
                    # this is doing sum(w * δij(i, phase)), where δij is the Kronecker delta
                    w = accumulate_weight(w, x, ph_local, NC)
                end
            end
            w = w .* inv(sum(w))
            for ip in cellaxes(ratio_midpoints)
                @index ratio_midpoints[ip, midpoint_index...] = w[ip] * !isnan(w[ip]) # make it zero if there are NaNs (means no particles within half cells)
            end
        end
    end

    return nothing
end

@inline function midpoint_offset(::Val{3}, dimension::Symbol)
    return offsets = if dimension === :xy
        (1, 1, 0)
    elseif dimension === :yz
        (0, 1, 1)
    elseif dimension === :xz
        (1, 0, 1)
    else
        throw("Unknown dimensions. Valid dimensions are :xy, :yz, :xz")
    end
end

function lastboundary_offset(offsets::NTuple{3}, I::NTuple{3}, ni::NTuple{3})
    @inline
    return Base.@ntuple 3 i -> @inbounds Int(ni[i] == (offsets[i] * I[i]))
end
