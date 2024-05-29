
import Base: Array

## Conversion of structs to CPU

@inline remove_parameters(::T) where {T} = Base.typename(T).wrapper

_backend(::Particles{CPUBackend}) = CPUBackend
_backend(::MarkerChain{CPUBackend}) = CPUBackend
_backend(::PassiveMarkers{CPUBackend}) = CPUBackend

Array(x::T) where {T<:AbstractParticles} = Array(_backend(x), x)
Array(::Nothing) = nothing
Array(::CPUBackend, x) = x

function Array(::AbstractBackend, x::T) where {T<:AbstractParticles}
    nfields = fieldcount(T)
    cpu_fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        Array(getfield(x, i))
    end
    T_clean = remove_parameters(x)
    return T_clean(cpu_fields...)
end