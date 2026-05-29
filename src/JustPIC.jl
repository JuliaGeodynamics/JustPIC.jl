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
    AMDGPUBackend

Backend tag for AMDGPU-based array allocation and execution paths.
"""
struct AMDGPUBackend <: AbstractBackend end

export TA

function CA end

"""
    TA()
    TA(backend)

Return the plain array type associated with `backend`.

For the CPU backend this is `Array`. Extension packages may add backend-specific
definitions for accelerator arrays.
"""
TA() = Array
TA(::Type{CPUBackend}) = Array

include("particles.jl")
export AbstractParticles, Particles, MarkerChain, MarkerSurface, PassiveMarkers, cell_index, cell_length

include("PhaseRatios/PhaseRatios.jl")
export nphases, numphases

include("Advection/types.jl")
export AbstractAdvectionIntegrator, Euler, RungeKutta2, RungeKutta4

include("JustPIC_CPU.jl")

include("CellArrays/conversion.jl")
export Array, copy

end # module
