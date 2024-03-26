include("types.jl")
export AbstractAdvectionIntegrator, Euler, RungeKutta2

include("Euler.jl")

include("RK2.jl")

include("advection.jl")
export advection!