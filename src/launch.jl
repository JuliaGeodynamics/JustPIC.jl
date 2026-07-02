# KernelAbstractions launch + allocation helpers.
#
# This is the compatibility layer that lets JustPIC kernels be written once with
# `@kernel` and launched on any backend. It replaces the old backend-global
# launcher and allocation machinery:
#
#   * `ka_backend(x)`      -> the KernelAbstractions backend of an array/container
#   * `launch!(...)`       -> synchronous kernel launch
#   * `cell_array(be, ...)`-> backend-aware `CellArray` allocation
#
# The GPU backends add methods for `ka_backend` / `undef_cell_array` in their
# package extensions; everything else here is backend-agnostic.

# Imported unqualified (not `using`) so KernelAbstractions' `@index` export does
# not clash with CellArraysIndexing's `@index`, which the kernels rely on.
import KernelAbstractions

# ---------------------------------------------------------------------------
# Backend accessors
# ---------------------------------------------------------------------------

@inline ka_backend(A::AbstractArray) = KernelAbstractions.get_backend(A)
@inline ka_backend(A::CellArray) = KernelAbstractions.get_backend(A.data)
@inline ka_backend(p::Particles) = KernelAbstractions.get_backend(p.index.data)
@inline ka_backend(p::MarkerChain) = KernelAbstractions.get_backend(p.index.data)
@inline ka_backend(p::PassiveMarkers) = KernelAbstractions.get_backend(p.coords[1])
@inline ka_backend(p::JustPIC.PhaseRatios) = KernelAbstractions.get_backend(p.center.data)

# ---------------------------------------------------------------------------
# Kernel launch
# ---------------------------------------------------------------------------

"""
    launch!(backend, kernel!, ndrange, args...)

Instantiate and run KernelAbstractions `kernel!` on `backend` over `ndrange`,
then block until it completes.

The trailing `synchronize` keeps the package's historical synchronous launch
semantics: host reads, MPI halo exchanges and injection/cleanup all assume the
previous kernel has finished. A later optimization pass may drop per-launch
synchronization in favor of synchronizing only before host access.
"""
@inline function launch!(backend, kernel!::F, ndrange, args::Vararg{Any, N}) where {F, N}
    kernel!(backend)(args...; ndrange = ndrange)
    KernelAbstractions.synchronize(backend)
    return nothing
end

# ---------------------------------------------------------------------------
# CellArray allocation
# ---------------------------------------------------------------------------

# Uninitialized `CellArray` of cell-type `T` on `backend`, using the
# backend-appropriate block length (1 == array-of-struct on the CPU,
# 0 == struct-of-array on the GPU). GPU backends define their own methods.
@inline function undef_cell_array(
        ::Type{CPUBackend}, ::Type{T}, ni::NTuple{N, <:Integer}
    ) where {T, N}
    return CPUCellArray{T, 1}(undef, ni)
end

"""
    cell_array(backend, x, ncells::NTuple, ni::NTuple)
    cell_array(x, ncells::NTuple, ni::NTuple)

Allocate a `CellArray` on `backend`, with `ncells` entries per grid cell over a
grid of size `ni`, and fill every entry with `x`.

The two-argument backend form is the preferred allocation path for particle
storage and phase-ratio arrays. The backend-less form allocates on
`CPUBackend`.

# Examples
```julia
index = cell_array(JustPIC.CPUBackend, false, (24,), (64, 64))
field = cell_array(JustPIC.CPUBackend, 0.0, (3,), (64, 64))
```
"""
@inline function cell_array(
        backend::Type{<:AbstractBackend}, x::T, ncells::NTuple{Nc, <:Integer}, ni::NTuple{Nd, <:Integer}
    ) where {T, Nc, Nd}
    celltype = SArray{Tuple{ncells...}, T, Nc, prod(ncells)}
    A = undef_cell_array(backend, celltype, ni)
    fill!(A.data, x)
    return A
end

@inline function cell_array(x::T, ncells::NTuple{Nc, <:Integer}, ni::NTuple{Nd, <:Integer}) where {T, Nc, Nd}
    return cell_array(CPUBackend, x, ncells, ni)
end
