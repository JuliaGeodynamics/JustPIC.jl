module JustPIC

using MPI: MPI
using ImplicitGlobalGrid
using MuladdMacro
using GridGeometryUtils
using CellArrays, StaticArrays

# `using KernelAbstractions` (full) makes bare `@index` resolve to KA's, which is
# what `@kernel` bodies need. It also brings the KA backends `CPU` and the abstract
# supertype `Backend`. The vendor backends `CUDA.CUDABackend`, `AMDGPU.ROCBackend`,
# `Metal.MetalBackend` are introduced by the respective package extensions. JustPIC
# no longer defines its own backend tags.
using KernelAbstractions

# CellArraysIndexing element access is used through the `CAI.@index` qualifier so it
# does not clash with KA's `@index`; the non-`@index` names are used bare.
using CellArraysIndexing: @cell, getcell, setcell!, getcellindex, setcellindex!
import CellArraysIndexing as CAI

export @cell
export TA, CA

"""
    TA()
    TA(backend)

Return the plain array type associated with `backend` (a KernelAbstractions
backend type such as `CPU`).

For `CPU` this is `Array`. Loading CUDA.jl / AMDGPU.jl / Metal.jl extends this for
`CUDA.CUDABackend`, `AMDGPU.ROCBackend`, and `Metal.MetalBackend`, respectively.
"""
TA() = Array
TA(::Type{CPU}) = Array

"""
    CA(backend, dims; eltype = Float64)

Allocate an uninitialized `CellArray` of size `dims` on `backend`. Extended for
the GPU backends by the package extensions.
"""
CA(::Type{CPU}, dims; eltype = Float64) = CPUCellArray{eltype}(undef, dims)

include("particles.jl")
export AbstractParticles, Particles, MarkerChain, PassiveMarkers, cell_index, cell_length

include("PhaseRatios/PhaseRatios.jl")
export nphases, numphases

include("Advection/types.jl")
export AbstractAdvectionIntegrator, Euler, RungeKutta2, RungeKutta4, set_precision

# The algorithm catalog. Since the KA kernels are dimension-generic, this is
# included once into `JustPIC` directly. Use `using JustPIC`.
include("common.jl")

include("CellArrays/conversion.jl")
export Array, copy

end # module
