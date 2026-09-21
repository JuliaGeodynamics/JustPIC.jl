using JustPIC, Test
import KernelAbstractions: CPU, get_backend

# Fails a suite that would run on another backend or precision than requested, e.g. a
# silent CPU fallback inside a GPU job. `backend` is the KernelAbstractions backend type.
function check_backend(name::AbstractString, backend, ::Type{FT}) where {FT}
    @testset "Backend selection ($name, $FT)" begin
        @test name in ("CPU", "CUDA", "AMDGPU", "Metal")

        precision = get(ENV, "JULIA_JUSTPIC_PRECISION", "")
        isempty(precision) || @test string(FT) == precision
        name == "Metal" && @test FT === Float32

        if name != "CPU"
            @test backend !== CPU
            @test getproperty(@__MODULE__, Symbol(name)).functional()
        end

        x = JustPIC.TA(backend)(zeros(FT, 2))
        @test get_backend(x) isa backend
        @test eltype(x) === FT

        cells = JustPIC.CA(backend, (2, 2); eltype = FT)
        @test JustPIC.ka_backend(cells) isa backend
        @test eltype(cells) === FT
    end
    return nothing
end

# Full legacy suites inspect device arrays on host. Allow scalar access only
# inside isolated full-suite processes; fast tier keeps it forbidden.
if get(ENV, "JULIA_JUSTPIC_ALLOW_SCALAR", "false") == "true"
    @static if isdefined(Main, :AMDGPU) && get(ENV, "JULIA_JUSTPIC_BACKEND", "CPU") == "AMDGPU"
        AMDGPU.allowscalar(true)
    elseif isdefined(Main, :CUDA) && get(ENV, "JULIA_JUSTPIC_BACKEND", "CPU") == "CUDA"
        CUDA.allowscalar(true)
    elseif isdefined(Main, :Metal) && get(ENV, "JULIA_JUSTPIC_BACKEND", "CPU") == "Metal"
        Metal.allowscalar(true)
    end
end
