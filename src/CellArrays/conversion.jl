import Base: Array, copy

@inline remove_parameters(::T) where {T} = Base.typename(T).wrapper

# detect if we are on the CPU (`Val{false}`) or GPU (`Val{true}`)
@inline isdevice(::Type{Array{T, N}}) where {T,N} = Val(false)
@inline isdevice(::Type{T}) where {T<:AbstractArray} = Val(true) # this is a big assumption but still
@inline isdevice(::T) where {T} =
    throw(ArgumentError("$(T) is not a supported CellArray type."))

@inline CPU_CellArray(::Type{T}, ::UndefInitializer, dims::NTuple{N,Int}) where {T<:CellArrays.Cell,N} = CellArrays.CellArray{T,N, 1, Array{eltype(T),3}}(undef, dims)
@inline CPU_CellArray(::Type{T}, ::UndefInitializer, dims::Int...) where {T<:CellArrays.Cell} = CPU_CellArray(T, undef, dims)


# Copies CellArray to CPU if it is on a GPU device
Array(CA::CellArray) = Array(isdevice(typeof(CA).parameters[end]), CA)
Array(::Val{false}, CA::CellArray) = CA

# inner kernel doing the actual copy of the `CellArray`
function Array(::Val{true}, CA::CellArray)
    dims = size(CA)
    T_SArray = first(typeof(CA).parameters)
    CA_cpu = CPU_CellArray(T_SArray, undef, dims)
    tmp = Array(CA.data)
    copyto!(CA_cpu.data, tmp)
    return CA_cpu
end

# recursively convert the data from `AbstractParticles` to CPU arrays
function Array(x::T) where {T<:AbstractParticles}
    nfields = fieldcount(T)
    cpu_fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        _Array(getfield(x, i))
    end
    T_clean = remove_parameters(x)
    return T_clean(CPUBackend, cpu_fields...)
end

_Array(x) = x
_Array(::Nothing) = nothing
_Array(x::AbstractArray) = Array(x)
_Array(x::NTuple{N,T}) where {N,T} = ntuple(i -> _Array(x[i]), Val(N))

# recursively copy the data from `AbstractParticles` to CPU arrays
function copy(x::T) where {T<:AbstractParticles}
    nfields = fieldcount(T)
    copied_fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        _copy(getfield(x, i))
    end
    T_clean = remove_parameters(x)
    return T_clean(copied_fields...)
end

_copy(::Nothing) = nothing
_copy(x::AbstractArray) = copy(x)
_copy(x::NTuple{N,T}) where {N,T} = ntuple(i -> _copy(x[i]), Val(N))
_copy(x::T) where {T} = x
