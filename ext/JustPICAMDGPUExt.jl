module JustPICAMDGPUExt

using AMDGPU
using JustPIC, CellArrays, StaticArrays

import JustPIC: Particles, MarkerChain
# `ROCBackend` is AMDGPU.jl's KernelAbstractions backend (JustPIC no longer defines
# its own backend tags — it dispatches on the KA backends directly).
using AMDGPU: ROCBackend

# ---------------------------------------------------------------------------
# Collapsed GPU extension
# ---------------------------------------------------------------------------
# JustPIC's kernels are backend-generic: they launch through
# `launch!(ka_backend(x), ...)` and pick the KernelAbstractions backend from the
# array type at runtime. This extension therefore only supplies the AMDGPU-specific
# *allocation* and *host <-> device conversion* primitives; the generic methods
# already compiled into `JustPIC` dispatch to AMDGPU
# automatically. No `common.jl` re-include and no per-function forwarding layer
# (`JustPIC.f(::Particles{ROCBackend}, ...) = f(...)`) are needed.

# `CellArrays.@define_ROCCellArray` is not used: it defines methods on CellArrays-owned
# types, which clash with identical definitions from any other package calling the
# same macro (e.g. ParallelStencil). See #306.
const ROCCellArray{T, N, B, T_elem} = CellArray{T, N, B, ROCArray{T_elem, CellArrays._N}}

# Uninitialized `ROCCellArray` with cell type `T` and array-of-structs layout (`B = 0`).
@inline function _roccellarray(::Type{T}, dims::NTuple{N, Integer}) where {T, N}
    return ROCCellArray{T, N, 0, eltype(T)}(undef, Int.(dims))
end

JustPIC.TA(::Type{ROCBackend}) = ROCArray

# ---------------------------------------------------------------------------
# Backend-specific CellArray allocation
# ---------------------------------------------------------------------------

JustPIC.CA(::Type{ROCBackend}, dims; eltype = Float64) = _roccellarray(eltype, dims)

@inline function JustPIC.undef_cell_array(
        ::Type{ROCBackend}, ::Type{T}, ni::NTuple{N, <:Integer}
    ) where {T, N}
    return _roccellarray(T, ni)
end

# ---------------------------------------------------------------------------
# Host -> device conversions
# ---------------------------------------------------------------------------

function AMDGPU.ROCArray(::Type{T}, particles::JustPIC.Particles) where {T <: Number}
    (; coords, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel) = particles
    coords_gpu = ntuple(i -> ROCArray(T, coords[i]), Val(length(coords)))
    di_gpu = (; center = map(x -> ROCArray(T, x), di.center), vertex = map(x -> ROCArray(T, x), di.vertex), velocity = map(vg -> map(x -> ROCArray(T, x), vg), di.velocity))
    _di_gpu = (; center = map(x -> ROCArray(T, x), _di.center), vertex = map(x -> ROCArray(T, x), _di.vertex), velocity = map(vg -> map(x -> ROCArray(T, x), vg), _di.velocity))
    xci_gpu = map(x -> ROCArray(T, x), xci)
    xvi_gpu = map(x -> ROCArray(T, x), xvi)
    xi_vel_gpu = map(vg -> map(x -> ROCArray(T, x), vg), xi_vel)
    return Particles(
        ROCBackend,
        coords_gpu,
        ROCArray(Bool, index),
        nxcell,
        max_xcell,
        min_xcell,
        np,
        di_gpu,
        _di_gpu,
        xci_gpu,
        xvi_gpu,
        xi_vel_gpu,
    )
end

function AMDGPU.ROCArray(::Type{T}, chain::JustPIC.MarkerChain) where {T <: Number}
    (;
        cell_vertices, coords, coords0, h_vertices, h_vertices0, index, max_xcell, min_xcell,
    ) = chain
    coords_gpu = ntuple(i -> ROCArray(T, coords[i]), Val(length(coords)))
    coords0_gpu = ntuple(i -> ROCArray(T, coords0[i]), Val(length(coords0)))
    return MarkerChain(
        ROCBackend,
        coords_gpu,
        coords0_gpu,
        ROCArray(h_vertices),
        ROCArray(h_vertices0),
        cell_vertices,
        ROCArray(Bool, index),
        max_xcell,
        min_xcell,
    )
end

function AMDGPU.ROCArray(::Type{T}, phase_ratios::JustPIC.PhaseRatios) where {T <: Number}
    (; center, vertex, Vx, Vy, Vz, yz, xz, xy) = phase_ratios
    return JustPIC.PhaseRatios(
        ROCBackend,
        ROCArray(T, center),
        ROCArray(T, vertex),
        ROCArray(T, Vx),
        ROCArray(T, Vy),
        ROCArray(T, Vz),
        ROCArray(T, yz),
        ROCArray(T, xz),
        ROCArray(T, xy),
    )
end

function AMDGPU.ROCArray(phase_ratios::JustPIC.PhaseRatios)
    (; center, vertex, Vx, Vy, Vz, yz, xz, xy) = phase_ratios
    return JustPIC.PhaseRatios(
        ROCBackend,
        ROCArray(center),
        ROCArray(vertex),
        ROCArray(Vx),
        ROCArray(Vy),
        ROCArray(Vz),
        ROCArray(yz),
        ROCArray(xz),
        ROCArray(xy),
    )
end

function AMDGPU.ROCArray(particles::JustPIC.Particles)
    (; coords, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel) = particles
    coords_gpu = ntuple(i -> ROCArray(coords[i]), Val(length(coords)))
    di_gpu = (; center = map(ROCArray, di.center), vertex = map(ROCArray, di.vertex), velocity = map(vg -> map(ROCArray, vg), di.velocity))
    _di_gpu = (; center = map(ROCArray, _di.center), vertex = map(ROCArray, _di.vertex), velocity = map(vg -> map(ROCArray, vg), _di.velocity))
    xci_gpu = map(ROCArray, xci)
    xvi_gpu = map(ROCArray, xvi)
    xi_vel_gpu = map(vg -> map(ROCArray, vg), xi_vel)
    return Particles(
        ROCBackend,
        coords_gpu,
        ROCArray(index),
        nxcell,
        max_xcell,
        min_xcell,
        np,
        di_gpu,
        _di_gpu,
        xci_gpu,
        xvi_gpu,
        xi_vel_gpu,
    )
end

function AMDGPU.ROCArray(chain::JustPIC.MarkerChain)
    (;
        cell_vertices, coords, coords0, h_vertices, h_vertices0, index, max_xcell, min_xcell,
    ) = chain
    coords_gpu = ntuple(i -> ROCArray(coords[i]), Val(length(coords)))
    coords0_gpu = ntuple(i -> ROCArray(coords0[i]), Val(length(coords0)))
    return MarkerChain(
        ROCBackend,
        coords_gpu,
        coords0_gpu,
        ROCArray(h_vertices),
        ROCArray(h_vertices0),
        cell_vertices,
        ROCArray(Bool, index),
        max_xcell,
        min_xcell,
    )
end

function AMDGPU.ROCArray(
        ::Type{T}, CA::CellArray{S, N, B}
    ) where {T <: Number, S, N, B}
    ni = size(CA)
    # Array initializations
    T_SArray = eltype(CA)
    CA_ROC = _roccellarray(SVector{length(T_SArray), T}, ni)
    # copy data to the ROC CellArray
    # CPU particle fields use B=1; ROC fields use B=0.
    tmp = if B == 0
        ROCArray(CA.data)
    else
        ROCArray(permutedims(CA.data, (3, 2, 1)))
    end
    copyto!(CA_ROC.data, tmp)
    return CA_ROC
end

AMDGPU.ROCArray(particles::JustPIC.Particles{ROCBackend}) = particles
AMDGPU.ROCArray(phase_ratios::JustPIC.PhaseRatios{ROCBackend}) = phase_ratios
AMDGPU.ROCArray(CA::CellArray) = AMDGPU.ROCArray(eltype(eltype(CA)), CA)
AMDGPU.ROCArray(::Type{Float64}, A::Vector{Float64}) = AMDGPU.ROCArray(A)
AMDGPU.ROCArray(::Type{T}, x::Number) where {T <: AbstractFloat} = x
AMDGPU.ROCArray(::Type{T}, x::AbstractRange) where {T <: AbstractFloat} = x
AMDGPU.ROCArray(x::T) where {T <: AbstractFloat} = x

# ---------------------------------------------------------------------------
# Positional reconstruction constructors (device-array `index`)
# ---------------------------------------------------------------------------
# GPU analogue of `Particles(coords, index::CPUCellArray, ...)` in `particles.jl`:
# reattach the `ROCBackend` tag when a `Particles` is rebuilt positionally from
# device arrays (e.g. checkpoint restore). Dispatched on the `index` cell-array
# dimension so 2D and 3D pick the right method.
function JustPIC.Particles(
        coords,
        index::CellArray{SVector{N1, Bool}, 2, 0, ROCArray{Bool, N2}},
        nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel,
    ) where {N1, N2}
    return Particles(ROCBackend, coords, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel)
end

function JustPIC.Particles(
        coords,
        index::CellArray{SVector{N1, Bool}, 2, 0, ROCArray{Bool, N2, B}},
        nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel,
    ) where {B, N1, N2}
    return Particles(ROCBackend, coords, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel)
end

function JustPIC.Particles(
        coords,
        index::CellArray{SVector{N1, Bool}, 3, 0, ROCArray{Bool, N2}},
        nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel,
    ) where {N1, N2}
    return Particles(ROCBackend, coords, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel)
end

function JustPIC.Particles(
        coords,
        index::CellArray{SVector{N1, Bool}, 3, 0, ROCArray{Bool, N2, B}},
        nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel,
    ) where {B, N1, N2}
    return Particles(ROCBackend, coords, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel)
end

end # module
