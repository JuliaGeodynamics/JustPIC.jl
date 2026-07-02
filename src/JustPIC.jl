module JustPIC

# using ImplicitGlobalGrid
using MPI: MPI
using CellArrays, CellArraysIndexing, StaticArrays

export @cell, @index

abstract type AbstractBackend end

"""
    CPUBackend

Backend tag for CPU array allocation and CPU execution paths.
"""
struct CPUBackend <: AbstractBackend end

"""
    CUDABackend

Backend tag for CUDA array allocation and CUDA execution paths.

CUDA-specific array and `CellArray` methods are installed by the CUDA package
extension when CUDA.jl is loaded.
"""
struct CUDABackend <: AbstractBackend end

"""
    AMDGPUBackend

Backend tag for AMDGPU array allocation and AMDGPU execution paths.

AMDGPU-specific array and `CellArray` methods are installed by the AMDGPU
package extension when AMDGPU.jl is loaded.
"""
struct AMDGPUBackend <: AbstractBackend end

"""
    MetalBackend

Backend tag for Metal (Apple GPU) array allocation and execution paths.

Metal-specific array and `CellArray` methods are installed by the Metal package
extension when Metal.jl is loaded.
"""
struct MetalBackend <: AbstractBackend end

export TA

function CA end

"""
    TA()
    TA(backend)

Return the plain array type associated with `backend`.

For `CPUBackend` this is `Array`. Loading CUDA.jl or AMDGPU.jl extends this for
`CUDABackend` and `AMDGPUBackend`, respectively.
"""
TA() = Array
TA(::Type{CPUBackend}) = Array

include("particles.jl")
export AbstractParticles, Particles, MarkerChain, PassiveMarkers, cell_index, cell_length

include("PhaseRatios/PhaseRatios.jl")
export nphases, numphases

include("Advection/types.jl")
export AbstractAdvectionIntegrator, Euler, RungeKutta2, RungeKutta4, set_precision

include("JustPIC_CPU.jl")

include("CellArrays/conversion.jl")
export Array, copy

end # module
