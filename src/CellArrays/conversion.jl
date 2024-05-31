import Base.Array

@inline isdevice(::Type{Array}) = Val(false)
@inline isdevice(::Type{T}) where {T<:AbstractArray} = Val(true) # this is a big assumption but still
@inline isdevice(::T) where T = throw(ArgumentError("$(T) is not a supported CellArray type."))

# Copies CellArray to CPU if it is on a GPU device
Array(CA::CellArray) = Array(isdevice(typeof(CA).parameters[end]), CA)
Array(::Val{false}, CA::CellArray) = CA

function Array(::Val{true}, CA::CellArray)
    dims     = size(CA)
    T_SArray = first(typeof(CA).parameters)
    CA_cpu   = CPUCellArray{T_SArray}(undef, dims)
    return CA_cpu
end

@inline remove_parameters(::T) where {T} = Base.typename(T).wrapper

# Array(x::T) where {T<:JR_T} = Array(backend(x), x)
_Array(::Nothing) = nothing
_Array(::T) where T = T
_Array(x::AbstractArray) = Array(x)
_Array(x::NTuple{N, T}) where {N, T} = ntuple(i-> _Array(x[i]), Val(N))

function Array(x::T) where {T<:AbstractParticles}
    nfields = fieldcount(T)
    cpu_fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        _Array(getfield(x, i))
    end
    T_clean = remove_parameters(x)
    return T_clean(cpu_fields...)
end