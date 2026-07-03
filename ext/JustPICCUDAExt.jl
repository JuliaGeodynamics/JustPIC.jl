module JustPICCUDAExt

using CUDA
using JustPIC, CellArrays, StaticArrays

import JustPIC: AbstractBackend, CUDABackend, Particles, MarkerChain

# ---------------------------------------------------------------------------
# Collapsed GPU extension
# ---------------------------------------------------------------------------
# JustPIC's kernels are backend-generic: they launch through
# `launch!(ka_backend(x), ...)` and pick the KernelAbstractions backend from the
# array type at runtime. This extension therefore only supplies the CUDA-specific
# *allocation* and *host <-> device conversion* primitives; the generic methods
# already compiled into `JustPIC._2D` / `JustPIC._3D` dispatch to CUDA
# automatically. No `common.jl` re-include and no per-function forwarding layer
# (`JustPIC._2D.f(::Particles{CUDABackend}, ...) = f(...)`) are needed.

CellArrays.@define_CuCellArray()

JustPIC.TA(::Type{CUDABackend}) = CuArray

function CuCellArray(
        ::Type{T}, ::UndefInitializer, dims::NTuple{N, Int}
    ) where {T <: CellArrays.Cell, N}
    return CellArrays.CellArray{T, N, 0, CUDA.CuArray{eltype(T), 3}}(undef, dims)
end
function CuCellArray(::Type{T}, ::UndefInitializer, dims::Int...) where {T <: CellArrays.Cell}
    return CuCellArray(T, undef, dims)
end

# ---------------------------------------------------------------------------
# Backend-specific CellArray allocation
# ---------------------------------------------------------------------------
# `CA` and `undef_cell_array` are per-dimension bindings (they live inside the
# `JustPIC._2D` / `JustPIC._3D` submodules because `launch.jl` is `include`d
# there), so both dimensions need a CUDA method.
for D in (JustPIC._2D, JustPIC._3D)
    @eval begin
        $D.CA(::Type{CUDABackend}, dims; eltype = Float64) =
            CuCellArray{eltype}(undef, dims)

        @inline function $D.undef_cell_array(
                ::Type{CUDABackend}, ::Type{T}, ni::NTuple{N, <:Integer}
            ) where {T, N}
            return CuCellArray{T}(undef, Int.(ni))
        end
    end
end

# ---------------------------------------------------------------------------
# Host -> device conversions
# ---------------------------------------------------------------------------

function CUDA.CuArray(::Type{T}, particles::JustPIC.Particles) where {T <: Number}
    (; coords, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel) = particles
    coords_gpu = ntuple(i -> CuArray(T, coords[i]), Val(length(coords)))
    di_gpu = (; center = map(x -> CuArray(T, x), di.center), vertex = map(x -> CuArray(T, x), di.vertex), velocity = map(vg -> map(x -> CuArray(T, x), vg), di.velocity))
    _di_gpu = (; center = map(x -> CuArray(T, x), _di.center), vertex = map(x -> CuArray(T, x), _di.vertex), velocity = map(vg -> map(x -> CuArray(T, x), vg), _di.velocity))
    xci_gpu = map(x -> CuArray(T, x), xci)
    xvi_gpu = map(x -> CuArray(T, x), xvi)
    xi_vel_gpu = map(vg -> map(x -> CuArray(T, x), vg), xi_vel)
    return Particles(
        CUDABackend,
        coords_gpu,
        CuArray(Bool, index),
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

function CUDA.CuArray(::Type{T}, chain::JustPIC.MarkerChain) where {T <: Number}
    (;
        cell_vertices, coords, coords0, h_vertices, h_vertices0, index, max_xcell, min_xcell,
    ) = chain
    coords_gpu = ntuple(i -> CuArray(T, coords[i]), Val(length(coords)))
    coords0_gpu = ntuple(i -> CuArray(T, coords0[i]), Val(length(coords0)))
    return MarkerChain(
        CUDABackend,
        coords_gpu,
        coords0_gpu,
        CuArray(h_vertices),
        CuArray(h_vertices0),
        cell_vertices,
        CuArray(Bool, index),
        max_xcell,
        min_xcell,
    )
end

function CUDA.CuArray(::Type{T}, phase_ratios::JustPIC.PhaseRatios) where {T <: Number}
    (; center, vertex, Vx, Vy, Vz, yz, xz, xy) = phase_ratios
    return JustPIC.PhaseRatios(
        CUDABackend,
        CuArray(T, center),
        CuArray(T, vertex),
        CuArray(T, Vx),
        CuArray(T, Vy),
        CuArray(T, Vz),
        CuArray(T, yz),
        CuArray(T, xz),
        CuArray(T, xy),
    )
end

function CUDA.CuArray(phase_ratios::JustPIC.PhaseRatios)
    (; center, vertex, Vx, Vy, Vz, yz, xz, xy) = phase_ratios
    return JustPIC.PhaseRatios(
        CUDABackend,
        CuArray(center),
        CuArray(vertex),
        CuArray(Vx),
        CuArray(Vy),
        CuArray(Vz),
        CuArray(yz),
        CuArray(xz),
        CuArray(xy),
    )
end

function CUDA.CuArray(particles::JustPIC.Particles)
    (; coords, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel) = particles
    coords_gpu = ntuple(i -> CuArray(coords[i]), Val(length(coords)))
    di_gpu = (; center = map(CuArray, di.center), vertex = map(CuArray, di.vertex), velocity = map(vg -> map(CuArray, vg), di.velocity))
    _di_gpu = (; center = map(CuArray, _di.center), vertex = map(CuArray, _di.vertex), velocity = map(vg -> map(CuArray, vg), _di.velocity))
    xci_gpu = map(CuArray, xci)
    xvi_gpu = map(CuArray, xvi)
    xi_vel_gpu = map(vg -> map(CuArray, vg), xi_vel)
    return Particles(
        CUDABackend,
        coords_gpu,
        CuArray(index),
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

function CUDA.CuArray(chain::JustPIC.MarkerChain)
    (;
        cell_vertices, coords, coords0, h_vertices, h_vertices0, index, max_xcell, min_xcell,
    ) = chain
    coords_gpu = ntuple(i -> CuArray(coords[i]), Val(length(coords)))
    coords0_gpu = ntuple(i -> CuArray(coords0[i]), Val(length(coords0)))
    return MarkerChain(
        CUDABackend,
        coords_gpu,
        coords0_gpu,
        CuArray(h_vertices),
        CuArray(h_vertices0),
        cell_vertices,
        CuArray(Bool, index),
        max_xcell,
        min_xcell,
    )
end

function CUDA.CuArray(::Type{T}, CA::CellArray) where {T <: Number}
    ni = size(CA)
    # Array initializations
    T_SArray = eltype(CA)
    CA_CUDA = CuCellArray(SVector{length(T_SArray), T}, undef, ni...)
    # copy data to the CUDA CellArray
    tmp = if size(CA.data) != size(CA_CUDA.data)
        CuArray(permutedims(CA.data, (3, 2, 1)))
    else
        CuArray(CA.data)
    end
    copyto!(CA_CUDA.data, tmp)
    return CA_CUDA
end

CUDA.CuArray(particles::JustPIC.Particles{CUDABackend}) = particles
CUDA.CuArray(phase_ratios::JustPIC.PhaseRatios{CUDABackend}) = phase_ratios
CUDA.CuArray(CA::CellArray) = CUDA.CuArray(eltype(eltype(CA)), CA)
CUDA.CuArray(::Type{T}, x::Number) where {T <: AbstractFloat} = x
CUDA.CuArray(::Type{T}, x::LinRange) where {T <: AbstractFloat} = x
CUDA.CuArray(x::T) where {T <: AbstractFloat} = x

# ---------------------------------------------------------------------------
# Positional reconstruction constructors (device-array `index`)
# ---------------------------------------------------------------------------
# GPU analogue of `Particles(coords, index::CPUCellArray, ...)` in `particles.jl`:
# reattach the `CUDABackend` tag when a `Particles` is rebuilt positionally from
# device arrays (e.g. checkpoint restore). Dispatched on the `index` cell-array
# dimension so 2D and 3D pick the right method.
function JustPIC.Particles(
        coords,
        index::CellArray{SVector{N1, Bool}, 2, 0, CuArray{Bool, N2}},
        nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel,
    ) where {N1, N2}
    return Particles(CUDABackend, coords, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel)
end

function JustPIC.Particles(
        coords,
        index::CellArray{SVector{N1, Bool}, 2, 0, CuArray{Bool, N2, B}},
        nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel,
    ) where {B, N1, N2}
    return Particles(CUDABackend, coords, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel)
end

function JustPIC.Particles(
        coords,
        index::CellArray{SVector{N1, Bool}, 3, 0, CuArray{Bool, N2}},
        nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel,
    ) where {N1, N2}
    return Particles(CUDABackend, coords, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel)
end

function JustPIC.Particles(
        coords,
        index::CellArray{SVector{N1, Bool}, 3, 0, CuArray{Bool, N2, B}},
        nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel,
    ) where {B, N1, N2}
    return Particles(CUDABackend, coords, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel)
end

end # module
