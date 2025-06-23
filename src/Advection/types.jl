abstract type AbstractAdvectionIntegrator end

struct Euler <: AbstractAdvectionIntegrator
    Euler(::Vararg{Any, N}) where {N} = new()
end

struct RungeKutta2{T} <: AbstractAdvectionIntegrator
    α::T

    function RungeKutta2(α::T) where {T}
        !(0 < α < 1) && throw(ArgumentError("Only 0 < α < 1 is supported"))
        return new{T}(α)
    end
end

RungeKutta2() = RungeKutta2(0.5)

struct RungeKutta4 <: AbstractAdvectionIntegrator
    RungeKutta4(::Vararg{Any, N}) where {N} = new()
end
