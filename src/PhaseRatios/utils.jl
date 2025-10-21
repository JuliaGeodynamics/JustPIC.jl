# 2D version, shear stress defined at cell vertices
function update_phase_ratios!(
        phase_ratios::JustPIC.PhaseRatios{B, T}, particles, xci, xvi, phases
    ) where {B, T <: AbstractMatrix}
    phase_ratios_center!(phase_ratios, particles, xci, phases)
    phase_ratios_vertex!(phase_ratios, particles, xvi, phases)
    # velocity nodes
    phase_ratios_face!(phase_ratios.Vx, particles, xci, phases, :x)
    phase_ratios_face!(phase_ratios.Vy, particles, xci, phases, :y)
    return nothing
end

# 3D version, shear stress defined at arete midpoints
function update_phase_ratios!(
        phase_ratios::JustPIC.PhaseRatios{B, T}, particles, xci, xvi, phases
    ) where {B, T <: AbstractArray}
    phase_ratios_center!(phase_ratios, particles, xci, phases)
    phase_ratios_vertex!(phase_ratios, particles, xvi, phases)
    # velocity nodes
    phase_ratios_face!(phase_ratios.Vx, particles, xci, phases, :x)
    phase_ratios_face!(phase_ratios.Vy, particles, xci, phases, :y)
    phase_ratios_face!(phase_ratios.Vz, particles, xci, phases, :z)
    # shear stress nodes
    phase_ratios_midpoint!(phase_ratios.xy, particles, xci, phases, :xy)
    phase_ratios_midpoint!(phase_ratios.yz, particles, xci, phases, :yz)
    phase_ratios_midpoint!(phase_ratios.xz, particles, xci, phases, :xz)
    return nothing
end

## interpolation kernels

function phase_ratio_weights(
        pxi::NTuple{NP, C}, ph::SVector{N1, T}, cell_center, di, ::Val{NC}
    ) where {N1, NC, NP, T, C}

    # Initiaze phase ratio weights
    w = ntuple(_ -> zero(T), Val(NC))

    for i in eachindex(ph)
        p = getindex.(pxi, i)
        isnan(first(p)) && continue
        x = @inline bilinear_weight(cell_center, p, di)
        ph_local = ph[i]
        # this is doing sum(w * δij(i, phase)), where δij is the Kronecker delta
        w = w .+ x .* ntuple(j -> (ph_local == j), Val(NC))
    end
    w = w .* inv(sum(w))
    return w
end

@generated function bilinear_weight(
        a::NTuple{N, T}, b::NTuple{N, T}, di::NTuple{N, T}
    ) where {N, T}
    return quote
        Base.@_inline_meta
        val = one($T)
        Base.Cartesian.@nexprs $N i ->
        @inbounds val *= muladd(-abs(a[i] - b[i]), inv(di[i]), one($T))
        return val
    end
end

## UTILS FOR MIDPOINTS AND FACES

@inline isinhalfcell(p, cell_midpoint, offsets, di) =
    prod(x -> abs(x[1] - x[2]) ≤ x[3] * (1 - x[4] / 2), zip(p, cell_midpoint, di, offsets))

@inline isinhalfcell(p, cell_midpoint, di) =
    prod(x -> abs(x[1] - x[2]) ≤ x[3] / 2, zip(p, cell_midpoint, di))

@inline accumulate_weight(w, x, phase, N) = w .+ x .* ntuple(j -> (phase == j), N)

@generated function isboundary(offsets::NTuple{N}, I::NTuple{N}) where {N}
    return quote
        @inline
        Base.@nany $N i -> @inbounds isone(offsets[i] * I[i])
    end
end
