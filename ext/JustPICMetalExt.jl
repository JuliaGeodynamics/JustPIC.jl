module JustPICMetalExt

using Metal
using JustPIC, CellArrays, StaticArrays

import JustPIC: AbstractBackend, MetalBackend

# ---------------------------------------------------------------------------
# Reference "collapsed" GPU extension
# ---------------------------------------------------------------------------
# Because JustPIC's kernels are now backend-generic (they launch through
# `launch!(ka_backend(x), ...)` and pick the KernelAbstractions backend from the
# array type at runtime), a GPU extension no longer needs to re-`include`
# `common.jl` into its own `_2D`/`_3D` submodules, nor a hand-written forwarding
# layer (`JustPIC._2D.f(::Particles{Backend}, ...) = f(...)`) per public function.
#
# All this extension supplies is the backend-specific *allocation* and
# *host <-> device conversion* primitives; the generic methods already compiled
# into `JustPIC._2D` / `JustPIC._3D` dispatch to Metal automatically.
#
# NOTE ON PRECISION: Apple Metal GPUs do not support `Float64`. Build containers
# on the CPU (`Float64`) and move them to the GPU with the eltype-typed
# conversions below, e.g. `particles = MtlArray(Float32, particles)`. The plain
# (untyped) conversions preserve the source eltype and therefore only work for
# already-`Float32` data.

CellArrays.@define_MtlCellArray()

JustPIC.TA(::Type{MetalBackend}) = MtlArray

# ---------------------------------------------------------------------------
# Backend-specific CellArray allocation
# ---------------------------------------------------------------------------
# `CA` and `undef_cell_array` are per-dimension bindings (they live inside the
# `JustPIC._2D` / `JustPIC._3D` submodules because `launch.jl` is `include`d
# there), so both need a Metal method.

for D in (JustPIC._2D, JustPIC._3D)
    @eval begin
        $D.CA(::Type{MetalBackend}, dims; eltype = Float32) =
            MtlCellArray{eltype}(undef, dims)

        @inline function $D.undef_cell_array(
                ::Type{MetalBackend}, ::Type{T}, ni::NTuple{N, <:Integer}
            ) where {T, N}
            return MtlCellArray{T}(undef, Int.(ni))
        end
    end
end

# ---------------------------------------------------------------------------
# Host -> device conversions
# ---------------------------------------------------------------------------

# Scalars / ranges are passed through unchanged (they live on the host and are
# forwarded to kernels by value).
Metal.MtlArray(::Type{T}, x::Number) where {T <: AbstractFloat} = x
Metal.MtlArray(::Type{T}, x::LinRange) where {T <: AbstractFloat} = x
Metal.MtlArray(x::T) where {T <: AbstractFloat} = x

# Plain host arrays (grid-spacing / coordinate vectors): narrow the eltype on the
# host before upload, since Metal rejects Float64.
Metal.MtlArray(::Type{T}, x::AbstractArray) where {T <: Number} = MtlArray(T.(collect(x)))

function Metal.MtlArray(::Type{T}, particles::JustPIC.Particles) where {T <: Number}
    (; coords, index, nxcell, max_xcell, min_xcell, np, di, _di, xci, xvi, xi_vel) = particles
    coords_gpu = ntuple(i -> MtlArray(T, coords[i]), Val(length(coords)))
    di_gpu = (; center = map(x -> MtlArray(T, x), di.center), vertex = map(x -> MtlArray(T, x), di.vertex), velocity = map(vg -> map(x -> MtlArray(T, x), vg), di.velocity))
    _di_gpu = (; center = map(x -> MtlArray(T, x), _di.center), vertex = map(x -> MtlArray(T, x), _di.vertex), velocity = map(vg -> map(x -> MtlArray(T, x), vg), _di.velocity))
    xci_gpu = map(x -> MtlArray(T, x), xci)
    xvi_gpu = map(x -> MtlArray(T, x), xvi)
    xi_vel_gpu = map(vg -> map(x -> MtlArray(T, x), vg), xi_vel)
    return Particles(
        MetalBackend,
        coords_gpu,
        MtlArray(Bool, index),
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

function Metal.MtlArray(::Type{T}, chain::JustPIC.MarkerChain) where {T <: Number}
    (;
        cell_vertices, coords, coords0, h_vertices, h_vertices0, index, max_xcell, min_xcell,
    ) = chain
    coords_gpu = ntuple(i -> MtlArray(T, coords[i]), Val(length(coords)))
    coords0_gpu = ntuple(i -> MtlArray(T, coords0[i]), Val(length(coords0)))
    return MarkerChain(
        MetalBackend,
        coords_gpu,
        coords0_gpu,
        MtlArray(T, h_vertices),
        MtlArray(T, h_vertices0),
        cell_vertices,
        MtlArray(Bool, index),
        max_xcell,
        min_xcell,
    )
end

function Metal.MtlArray(::Type{T}, phase_ratios::JustPIC.PhaseRatios) where {T <: Number}
    (; center, vertex, Vx, Vy, Vz, yz, xz, xy) = phase_ratios
    return JustPIC.PhaseRatios(
        MetalBackend,
        MtlArray(T, center),
        MtlArray(T, vertex),
        MtlArray(T, Vx),
        MtlArray(T, Vy),
        MtlArray(T, Vz),
        MtlArray(T, yz),
        MtlArray(T, xz),
        MtlArray(T, xy),
    )
end

function Metal.MtlArray(::Type{T}, CA::CellArray) where {T <: Number}
    ni = size(CA)
    T_SArray = eltype(CA)
    CA_Mtl = MtlCellArray{SVector{length(T_SArray), T}}(undef, ni)
    # Narrow the eltype on the host first: Metal has no Float64, so the source
    # Float64 backing array cannot be uploaded as-is (unlike CUDA/AMDGPU).
    host = if size(CA.data) != size(CA_Mtl.data)
        # transpose array-of-struct (CPU) layout to struct-of-array (GPU) layout
        permutedims(CA.data, (3, 2, 1))
    else
        CA.data
    end
    copyto!(CA_Mtl.data, MtlArray(T.(host)))
    return CA_Mtl
end

# Untyped conversions preserve the source eltype (only valid for Float32 data on Metal).
Metal.MtlArray(particles::JustPIC.Particles) = MtlArray(eltype(particles.coords[1].data), particles)
Metal.MtlArray(chain::JustPIC.MarkerChain) = MtlArray(eltype(chain.coords[1].data), chain)
Metal.MtlArray(phase_ratios::JustPIC.PhaseRatios) = MtlArray(eltype(phase_ratios.center.data), phase_ratios)
Metal.MtlArray(CA::CellArray) = MtlArray(eltype(eltype(CA)), CA)

# Already on the Metal backend: identity.
Metal.MtlArray(particles::JustPIC.Particles{MetalBackend}) = particles
Metal.MtlArray(chain::JustPIC.MarkerChain{MetalBackend}) = chain
Metal.MtlArray(phase_ratios::JustPIC.PhaseRatios{MetalBackend}) = phase_ratios

end # module
