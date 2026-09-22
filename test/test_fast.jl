const BACKEND_NAME = get(ENV, "JULIA_JUSTPIC_BACKEND", "CPU")

@static if BACKEND_NAME == "AMDGPU"
    using AMDGPU
elseif BACKEND_NAME == "CUDA"
    using CUDA
elseif BACKEND_NAME == "Metal"
    using Metal
end

using JustPIC, Test
import KernelAbstractions: CPU

const backend = @static if BACKEND_NAME == "AMDGPU"
    AMDGPU.ROCBackend
elseif BACKEND_NAME == "CUDA"
    CUDA.CUDABackend
elseif BACKEND_NAME == "Metal"
    Metal.MetalBackend
else
    CPU
end
const FT = if BACKEND_NAME == "Metal" || get(ENV, "JULIA_JUSTPIC_PRECISION", "") == "Float32"
    Float32
else
    Float64
end
include("helpers_backend.jl")
check_backend(BACKEND_NAME, backend, FT)

@static if BACKEND_NAME == "AMDGPU"
    AMDGPU.allowscalar(false)
elseif BACKEND_NAME == "CUDA"
    CUDA.allowscalar(false)
elseif BACKEND_NAME == "Metal"
    Metal.allowscalar(false)
end

@testset "Fast deterministic correctness" begin
    rk2 = RungeKutta2()
    p = (FT(1), FT(2))
    v0 = (FT(3), FT(4))
    v1 = (FT(3.1), FT(4.1))
    p1 = JustPIC.first_stage(rk2, FT(0.1), v0, p)
    @test all(p1 .≈ (FT(1.15), FT(2.2)))
    @test all(JustPIC.second_stage(rk2, FT(0.1), v0, v1, p1) .≈ (FT(1.46), FT(2.61)))

    values = JustPIC.TA(backend)(FT[1, 2, 3])
    @test Array(values) == FT[1, 2, 3]

    markers = init_passive_markers(backend, (FT[1, 2], FT[3, 4]))
    @test Array(markers).coords == (FT[1, 2], FT[3, 4])
end
