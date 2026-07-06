"""
    AbstractAdvectionIntegrator

Abstract supertype for time integrators used by particle, passive-marker, and
marker-chain advection routines.
"""
abstract type AbstractAdvectionIntegrator end

"""
    Euler()

Forward-Euler advection integrator.

This is the cheapest available integrator and is mainly useful for simple tests
or when first-order accuracy is sufficient.
"""
struct Euler <: AbstractAdvectionIntegrator
    Euler(::Vararg{Any, N}) where {N} = new()
end

"""
    RungeKutta2(α = 0.5)

Second-order Runge-Kutta advection integrator.

The parameter `α` controls the intermediate stage location and must satisfy
`0 < α < 1`. The default `α = 0.5` corresponds to the midpoint method.
"""
struct RungeKutta2{T} <: AbstractAdvectionIntegrator
    α::T

    function RungeKutta2(α::T) where {T}
        if !(0 < α < 1)
            throw(ArgumentError("Only 0 < α < 1 is supported"))
        end
        return new{T}(α)
    end
end

RungeKutta2() = RungeKutta2(0.5)

"""
    RungeKutta4()

Classical fourth-order Runge-Kutta advection integrator.
"""
struct RungeKutta4 <: AbstractAdvectionIntegrator
    RungeKutta4(::Vararg{Any, N}) where {N} = new()
end
