import Base: Array, copy

@inline remove_parameters(::T) where {T} = Base.typename(T).wrapper

# detect if we are on the CPU (`Val{false}`) or GPU (`Val{true}`)
@inline isdevice(::Type{Array{T, N}}) where {T, N} = Val(false)
@inline isdevice(::Type{T}) where {T <: AbstractArray} = Val(true) # this is a big assumption but still
@inline isdevice(::T) where {T} =
    throw(ArgumentError("$(T) is not a supported CellArray type."))

@inline CPU_CellArray(
    ::Type{T}, ::UndefInitializer, dims::NTuple{N, Int}
) where {T <: CellArrays.Cell, N} = CellArrays.CellArray{T, N, 1, Array{eltype(T), 3}}(undef, dims)
@inline CPU_CellArray(
    ::Type{T}, ::UndefInitializer, dims::Int...
) where {T <: CellArrays.Cell} = CPU_CellArray(T, undef, dims)

# Copies CellArray to CPU if it is on a GPU device.
to_cpu(CA::CellArray) = to_cpu(eltype(eltype(CA)), CA)
to_cpu(A::AbstractArray) = Array(A)
to_cpu(::Type{T}, A::AbstractArray) where {T <: Number} = Array{T}(A)
to_cpu(x::NTuple{N}) where {N} = ntuple(i -> to_cpu(x[i]), Val(N))
to_cpu(::Type{T}, x::NTuple{N}) where {T <: Number, N} =
    ntuple(i -> to_cpu(T, x[i]), Val(N))
function to_cpu(::Type{T}, CA::CellArray) where {T <: Number}
    return _to_cpu_cellarray(isdevice(typeof(CA).parameters[end]), T, CA)
end
function _to_cpu(::Type{T}, CA::CellArray) where {T <: Number}
    return to_cpu(T, CA)
end
function _to_cpu(::Type{T}, A::AbstractArray) where {T <: Number}
    return Array{T}(A)
end
function _to_cpu(CA::CellArray)
    return to_cpu(CA)
end
function _to_cpu(A::AbstractArray)
    return Array(A)
end
function _to_cpu_cellarray(::Val{false}, ::Type{T}, CA::CellArray) where {T <: Number}
    T === eltype(eltype(CA)) && return CA
    return _to_cpu_cellarray(Val(true), T, CA)
end

# inner kernel doing the actual copy of the `CellArray`
function _to_cpu_cellarray(::Val{true}, ::Type{T}, CA::CellArray) where {T <: Number}
    dims = size(CA)
    T_SArray = eltype(CA)
    CA_cpu = CPU_CellArray(SVector{length(T_SArray), T}, undef, dims)
    tmp = if size(CA.data) != size(CA_cpu.data)
        Array(permutedims(CA.data, (3, 2, 1)))
    else
        Array(CA.data)
    end
    copyto!(CA_cpu.data, T.(tmp))
    return CA_cpu
end

# recursively convert the data from `AbstractParticles` to CPU arrays
function Array(::Type{T}, x::P) where {T <: Number, P <: AbstractParticles}
    nfields = fieldcount(P)
    cpu_fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        if fieldname(P, i) === :index
            _to_cpu(Bool, getfield(x, i))
        else
            _to_cpu(T, getfield(x, i))
        end
    end
    T_clean = remove_parameters(x)
    return T_clean(CPU, cpu_fields...)
end

function Array(::Type{T}, x::PassiveMarkers) where {T <: Number}
    return PassiveMarkers(CPU, _to_cpu(T, x.coords))
end

function Array(x::P) where {P <: AbstractParticles}
    nfields = fieldcount(P)
    cpu_fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        A = getfield(x, i)
        _to_cpu(A)
    end
    T_clean = remove_parameters(x)
    return T_clean(CPU, cpu_fields...)
end

function Array(x::PassiveMarkers)
    return PassiveMarkers(CPU, _to_cpu(x.coords))
end
# Array(x::T) where {T<:AbstractParticles} = Array(Float64, x)

_to_cpu(x) = x
_to_cpu(::Nothing) = nothing
_to_cpu(x::NTuple{N, T}) where {N, T} = ntuple(i -> _to_cpu(x[i]), Val(N))
_to_cpu(::Type{T}, ::Nothing) where {T <: Number} = nothing
_to_cpu(::Type{T}, x) where {T <: Number} = x
function _to_cpu(::Type{T}, x::NTuple{N, TA}) where {T <: Number, N, TA}
    return ntuple(i -> _to_cpu(T, x[i]), Val(N))
end

# recursively deep-copy an `AbstractParticles`, keeping every field on its current
# device. The `Backend` type parameter (always first) is passed back to the
# constructor so the copy stays on the same backend; a GPU container has no
# backend-free constructor to fall back on.
function copy(x::T) where {T <: AbstractParticles}
    nfields = fieldcount(T)
    copied_fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        _copy(getfield(x, i))
    end
    T_clean = remove_parameters(x)
    return T_clean(T.parameters[1], copied_fields...)
end

function copy(x::PassiveMarkers{B}) where {B}
    return PassiveMarkers(B, _copy(x.coords))
end

_copy(::Nothing) = nothing
_copy(x::AbstractArray) = copy(x)
_copy(x::NTuple{N, T}) where {N, T} = ntuple(i -> _copy(x[i]), Val(N))
_copy(x::T) where {T} = x
