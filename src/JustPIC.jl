module JustPIC

# using ImplicitGlobalGrid
using MPI: MPI
using CellArrays, CellArraysIndexing, StaticArrays

export @cell, @index

abstract type AbstractBackend end
struct CPUBackend <: AbstractBackend end
struct AMDGPUBackend <: AbstractBackend end

export TA

function CA end

TA() = Array
TA(::Type{CPUBackend}) = Array

include("particles.jl")
export AbstractParticles, Particles, MarkerChain, PassiveMarkers, cell_index, cell_length

include("PhaseRatios/PhaseRatios.jl")
export nphases, numphases

include("Advection/types.jl")
export AbstractAdvectionIntegrator, Euler, RungeKutta2, RungeKutta4

include("JustPIC_CPU.jl")

include("CellArrays/conversion.jl")
export Array, copy

end # module
