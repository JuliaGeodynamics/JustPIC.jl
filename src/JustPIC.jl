module JustPIC

export CPUBackend, CUDABackend, AMDGPUBackend

abstract type AbstractBackend end
struct CPUBackend <: AbstractBackend end
struct AMDGPUBackend <: AbstractBackend end

export TA

function CA end

TA() = Array
TA(::Type{CPUBackend}) = Array

include("particles.jl")
export AbstractParticles, Particles, MarkerChain, PassiveMarkers, cell_index, cell_length

include("JustPIC_CPU.jl")

end # module
