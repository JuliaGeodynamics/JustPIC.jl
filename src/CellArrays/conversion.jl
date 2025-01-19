import Base: Array, copy

@inline remove_parameters(::T) where {T} = Base.typename(T).wrapper

# detect if we are on the CPU (`Val{false}`) or GPU (`Val{true}`)
@inline isdevice(::Type{Array{T, N}}) where {T, N} = Val(false)
@inline isdevice(::Type{T}) where {T <: AbstractArray} = Val(true) # this is a big assumption but still
@inline isdevice(::T) where {T} =
    throw(ArgumentError("$(T) is not a supported CellArray type."))

@inline CPU_CellArray(::Type{T}, ::UndefInitializer, dims::NTuple{N, Int}) where {T <: CellArrays.Cell, N} = CellArrays.CellArray{T, N, 1, Array{eltype(T), 3}}(undef, dims)
@inline CPU_CellArray(::Type{T}, ::UndefInitializer, dims::Int...) where {T <: CellArrays.Cell} = CPU_CellArray(T, undef, dims)


# Copies CellArray to CPU if it is on a GPU device
Array(CA::CellArray) = Array(eltype(eltype(CA)), CA)
Array(::Type{T}, CA::CellArray) where {T <: Number} = Array(isdevice(typeof(CA).parameters[end]), T, CA)
Array(::Val{false}, ::Type{T}, CA::CellArray) where {T <: Number} = Array(Val(true), T, CA)
Array(::Val{false}, ::Type{T}, CA::CellArray{CPUCellArray{SVector{N, T}}}) where {N, T <: Number} = CA

# inner kernel doing the actual copy of the `CellArray`
function Array(::Val{true}, ::Type{T}, CA::CellArray) where {T <: Number}
    dims = size(CA)
    T_SArray = eltype(CA)
    CA_cpu = CPU_CellArray(SVector{length(T_SArray), T}, undef, dims)
    tmp = if size(CA.data) != size(CA_cpu.data)
        Array(permutedims(CA.data, (3, 2, 1)))
    else
        Array(CA.data)
    end
    copyto!(CA_cpu.data, tmp)
    return CA_cpu
end

# recursively convert the data from `AbstractParticles` to CPU arrays
function Array(::Type{T}, x::P) where {T <: Number, P <: AbstractParticles}
    nfields = fieldcount(P)
    cpu_fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        if fieldname(P, i) === :index
            _Array(Bool, getfield(x, i))
        else
            _Array(T, getfield(x, i))
        end
    end
    T_clean = remove_parameters(x)
    return T_clean(CPUBackend, cpu_fields...)
end

function Array(x::P) where {P <: AbstractParticles}
    nfields = fieldcount(P)
    cpu_fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        A = getfield(x, i)
        _Array(A)
    end
    T_clean = remove_parameters(x)
    return T_clean(CPUBackend, cpu_fields...)
end
# Array(x::T) where {T<:AbstractParticles} = Array(Float64, x)

_Array(x) = x
_Array(::Nothing) = nothing
_Array(x::AbstractArray) = Array(x)
_Array(x::NTuple{N, T}) where {N, T} = ntuple(i -> _Array(x[i]), Val(N))
_Array(::Type{T}, ::Nothing) where {T <: Number} = nothing
_Array(::Type{T}, x) where {T <: Number} = x
_Array(::Type{T}, x::AbstractArray{TA, N}) where {T <: Number, N, TA} = Array(T, x)
_Array(::Type{T}, x::NTuple{N, TA}) where {T <: Number, N, TA} = ntuple(i -> _Array(T, x[i]), Val(N))

# recursively copy the data from `AbstractParticles` to CPU arrays
function copy(x::T) where {T <: AbstractParticles}
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
_copy(x::NTuple{N, T}) where {N, T} = ntuple(i -> _copy(x[i]), Val(N))
_copy(x::T) where {T} = x
