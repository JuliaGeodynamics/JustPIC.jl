struct PhaseRatios{Backend, T}
    center::T
    vertex::T

    function PhaseRatios(::Type{T}, ::Type{B}, nphases::Integer, ni::NTuple{N,Integer}) where {N, T, B}

        center = cell_array(0.0, (nphases, ), ni)
        vertex = cell_array(0.0, (nphases, ), ni.+1)

        new{B, typeof(center)}(center, vertex)
    end
end

PhaseRatios(nphases::Integer, ni::NTuple{N,Integer}) where {N} = PhaseRatios(Float64, CPUBackend, nphases, ni)
PhaseRatios(::Type{B}, nphases::Integer, ni::NTuple{N,Integer}) where {N, B<:AbstractBackend} = PhaseRatios(Float64, B, nphases, ni)

"""
    nphases(x::PhaseRatios)

Return the number of phases in `x::PhaseRatios`.
"""
@inline nphases(x::PhaseRatios) = nphases(x.center)
@inline numphases(x::PhaseRatios) = numphases(x.center)

@inline function nphases(
    ::CellArray{StaticArraysCore.SArray{Tuple{N},T,N1,N}}
) where {N,T,N1}
    return Val(N)
end

@inline function numphases(
    ::CellArray{StaticArraysCore.SArray{Tuple{N},T,N1,N}}
) where {N,T,N1}
    return N
end

## Kernels to compute phase ratios at the centers

function phase_ratios_center!(
    phase_ratios::PhaseRatios, particles, xci, phases
)
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

function phase_ratios_vertex!(
    phase_ratios::PhaseRatios, particles,xvi, phases
)
    ni = size(phases) .+ 1
    di = compute_dx(xvi)

    @parallel (@idx ni) phase_ratios_vertex_kernel!(
        phase_ratios.vertex, particles.coords, xvi, di, phases
    )
    return nothing
end

@parallel_indices (I...) function phase_ratios_vertex_kernel!(
    ratio_vertices, pxi::NTuple{3}, xvi::NTuple{3}, di::NTuple{3}, phases
)

    # # index corresponding to the cell center
    cell_vertex = ntuple(i -> xvi[i][I[i]], Val(3))
    ni = size(phases)

    for offsetᵢ in -1:0, offsetⱼ in -1:0, offsetₖ in -1:0
        offsets = offsetᵢ, offsetⱼ, offsetₖ
        cell_index = ntuple(Val(3)) do i
            clamp(I[i] + offsets[i], 1, ni[i])
        end
        # phase ratios weights (∑w = 1.0)
        w = phase_ratio_weights(
            getindex.(pxi, cell_index...),
            phases[cell_index...],
            cell_vertex,
            di,
            nphases(ratio_vertices),
        )
        # update phase ratios array
        for k in 1:numphases(ratio_vertices)
            @index ratio_vertices[k, I...] = w[k]
        end
    end

    return nothing
end

@parallel_indices (I...) function phase_ratios_vertex_kernel!(
    ratio_vertices, pxi::NTuple{2}, xvi::NTuple{2}, di::NTuple{2}, phases
)

    # index corresponding to the cell center
    cell_vertex = ntuple(i -> xvi[i][I[i]], Val(2))
    ni = size(phases)

    for offsetᵢ in -1:0, offsetⱼ in -1:0
        offsets = offsetᵢ, offsetⱼ
        cell_index = ntuple(Val(2)) do i
            clamp(I[i] + offsets[i], 1, ni[i])
        end
        # phase ratios weights (∑w = 1.0)
        w = phase_ratio_weights(
            getindex.(pxi, cell_index...),
            phases[cell_index...],
            cell_vertex,
            di,
            nphases(ratio_vertices),
        )
        # update phase ratios array
        for k in 1:numphases(ratio_vertices)
            @index ratio_vertices[k, I...] = w[k]
        end
    end

    return nothing
end


## interpolation kernels

function phase_ratio_weights(
    pxi::NTuple{NP,C}, ph::SVector{N1,T}, cell_center, di, ::Val{NC}
) where {N1,NC,NP,T,C}

    # Initiaze phase ratio weights (note: can't use ntuple() here because of the @generated function)
    w = ntuple(_ -> zero(T), Val(NC))
    sumw = zero(T)

    for i in eachindex(ph)
        p = getindex.(pxi, i)
        isnan(first(p)) && continue
        x = @inline bilinear_weight(cell_center, p, di)
        sumw += x # reduce
        ph_local = ph[i]
        # this is doing sum(w * δij(i, phase)), where δij is the Kronecker delta
        w = w .+ x .* ntuple(j -> (ph_local == j), Val(NC))
    end
    w = w .* inv(sumw)
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
