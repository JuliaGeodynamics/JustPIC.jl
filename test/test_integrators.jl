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

@testset "Runge-Kutta 2" begin
    @test RungeKutta2().α == 0.5
    @test RungeKutta2(2 / 3).α == 2 / 3
    @test_throws ArgumentError RungeKutta2(1.1)
    @test_throws ArgumentError RungeKutta2(-0.1)

    dt = 0.1
    p = 1.0, 2.0, 3.0
    v0 = 3.0, 4.0, 5.0
    v1 = 3.1, 4.1, 5.1

    rk2 = RungeKutta2()
    p1 = JustPIC.first_stage(rk2, dt, v0, p)
    @test p1 == (1.15, 2.2, 3.25)
    @test all(JustPIC.second_stage(rk2, dt, v0, v1, p1) .≈ (1.46, 2.61, 3.76))
    # stage updates act componentwise, so a 2D tuple matches the first two components
    @test JustPIC.first_stage(rk2, dt, v0[1:2], p[1:2]) == p1[1:2]

    rk2 = RungeKutta2(2 / 3)
    p1 = JustPIC.first_stage(rk2, dt, v0, p)
    @test all(p1 .≈ (1.2, 2.266666666, 3.3333333333333335))
    @test all(JustPIC.second_stage(rk2, dt, v0, v1, p1) .≈ (1.5075, 2.67416666, 3.8408333333333333))
end

@testset "Euler" begin
    @test Euler() isa Euler
    @test Euler(1) isa Euler

    p1 = JustPIC.first_stage(Euler(), 0.1, (3.0, 4.0, 5.0), (1.0, 2.0, 3.0))
    @test p1 == (1.3, 2.4, 3.5)
    @test JustPIC.first_stage(Euler(), 0.1, (3.0, 4.0), (1.0, 2.0)) == p1[1:2]
end
