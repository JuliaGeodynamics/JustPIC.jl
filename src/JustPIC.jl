module JustPIC

__precompile__(false)

export CPUBackend, CUDABackend, AMDGPUBackend

abstract type AbstractBackend end
struct CPUBackend <: AbstractBackend end
# struct CUDABackend <: AbstractBackend end
# struct AMDGPUBackend <: AbstractBackend end

export TA

TA() = Array
TA(::Type{CPUBackend}) = Array

include("Particles/particles.jl")
export Particles

include("JustPIC_CPU.jl")

end # module
