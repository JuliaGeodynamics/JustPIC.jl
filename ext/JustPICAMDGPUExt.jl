module JustPICAMDGPUExt

using AMDGPU
using JustPIC, CellArrays, StaticArrays

import JustPIC: AbstractBackend, AMDGPUBackend, Particles, MarkerChain

# ---------------------------------------------------------------------------
# Collapsed GPU extension
# ---------------------------------------------------------------------------
# JustPIC's kernels are backend-generic: they launch through
# `launch!(ka_backend(x), ...)` and pick the KernelAbstractions backend from the
# array type at runtime. This extension therefore only supplies the AMDGPU-specific
# *allocation* and *host <-> device conversion* primitives; the generic methods
# already compiled into `JustPIC._2D` / `JustPIC._3D` dispatch to AMDGPU
# automatically. No `common.jl` re-include and no per-function forwarding layer
# (`JustPIC._2D.f(::Particles{AMDGPUBackend}, ...) = f(...)`) are needed.

CellArrays.@define_ROCCellArray()

JustPIC.TA(::Type{AMDGPUBackend}) = ROCArray

function ROCCellArray(
        ::Type{T}, ::UndefInitializer, dims::NTuple{N, Int}
    ) where {T <: CellArrays.Cell, N}
    return CellArrays.CellArray{T, N, 0, AMDGPU.ROCArray{eltype(T), 3}}(undef, dims)
end
function ROCCellArray(
        ::Type{T}, ::UndefInitializer, dims::Int...
    ) where {T <: CellArrays.Cell}
    return ROCCellArray(T, undef, dims)
end

# ---------------------------------------------------------------------------
# Backend-specific CellArray allocation
# ---------------------------------------------------------------------------
# `CA` and `undef_cell_array` are per-dimension bindings (they live inside the
# `JustPIC._2D` / `JustPIC._3D` submodules because `launch.jl` is `include`d
# there), so both dimensions need an AMDGPU method.
for D in (JustPIC._2D, JustPIC._3D)
    @eval begin
        $D.CA(::Type{AMDGPUBackend}, dims; eltype = Float64) =
            ROCCellArray{eltype}(undef, dims)

        @inline function $D.undef_cell_array(
                ::Type{AMDGPUBackend}, ::Type{T}, ni::NTuple{N, <:Integer}
            ) where {T, N}
            return ROCCellArray{T}(undef, Int.(ni))
        end
    end
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
        AMDGPUBackend,
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
        AMDGPUBackend,
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
        AMDGPUBackend,
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
        AMDGPUBackend,
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
        AMDGPUBackend,
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
        AMDGPUBackend,
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

function AMDGPU.ROCArray(::Type{T}, CA::CellArray) where {T <: Number}
    ni = size(CA)
    # Array initializations
    T_SArray = eltype(CA)
    CA_ROC = ROCCellArray(SVector{length(T_SArray), T}, undef, ni...)
    # copy data to the ROC CellArray
    tmp = if size(CA.data) != size(CA_ROC.data)
        ROCArray(permutedims(CA.data, (3, 2, 1)))
    else
        ROCArray(CA.data)
    end
    copyto!(CA_ROC.data, tmp)
    return CA_ROC
end

AMDGPU.ROCArray(particles::JustPIC.Particles{AMDGPUBackend}) = particles
AMDGPU.ROCArray(phase_ratios::JustPIC.PhaseRatios{AMDGPUBackend}) = phase_ratios
AMDGPU.ROCArray(CA::CellArray) = AMDGPU.ROCArray(eltype(eltype(CA)), CA)
AMDGPU.ROCArray(::Type{Float64}, A::Vector{Float64}) = AMDGPU.ROCArray(A)
AMDGPU.ROCArray(::Type{T}, x::Number) where {T <: AbstractFloat} = x
AMDGPU.ROCArray(::Type{T}, x::AbstractRange) where {T <: AbstractFloat} = x
AMDGPU.ROCArray(x::T) where {T <: AbstractFloat} = x

# ---------------------------------------------------------------------------
# Positional reconstruction constructors (device-array `index`)
# ---------------------------------------------------------------------------
# GPU analogue of `Particles(coords, index::CPUCellArray, ...)` in `particles.jl`:
# reattach the `AMDGPUBackend` tag when a `Particles` is rebuilt positionally from
# device arrays (e.g. checkpoint restore). Dispatched on the `index` cell-array
# dimension so 2D and 3D pick the right method.
function JustPIC.Particles(
        coords,
        index::CellArray{SVector{N1, Bool}, 2, 0, ROCArray{Bool, N2}},
        nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel,
    ) where {N1, N2}
    return Particles(AMDGPUBackend, coords, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel)
end

function JustPIC.Particles(
        coords,
        index::CellArray{SVector{N1, Bool}, 2, 0, ROCArray{Bool, N2, B}},
        nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel,
    ) where {B, N1, N2}
    return Particles(AMDGPUBackend, coords, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel)
end

function JustPIC.Particles(
        coords,
        index::CellArray{SVector{N1, Bool}, 3, 0, ROCArray{Bool, N2}},
        nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel,
    ) where {N1, N2}
    return Particles(AMDGPUBackend, coords, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel)
end

function JustPIC.Particles(
        coords,
        index::CellArray{SVector{N1, Bool}, 3, 0, ROCArray{Bool, N2, B}},
        nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel,
    ) where {B, N1, N2}
    return Particles(AMDGPUBackend, coords, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel)
end

end # module
