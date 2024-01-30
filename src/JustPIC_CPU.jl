module _2D
using ImplicitGlobalGrid
using MPI: MPI
using MuladdMacro
using ParallelStencil
using CellArrays
using ..JustPIC

__precompile__(false)

@init_parallel_stencil(Threads, Float64, 2)
include("Includes.jl")
end

module _3D
using ImplicitGlobalGrid
using MPI: MPI
using MuladdMacro
using ParallelStencil
using CellArrays
using ..JustPIC

__precompile__(false)

@init_parallel_stencil(Threads, Float64, 3)
include("Includes.jl")
end
