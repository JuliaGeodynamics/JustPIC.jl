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
import KernelAbstractions: @kernel, @index

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

@kernel function rk2_stage_kernel!(result, α)
    i = @index(Global, Linear)
    T = eltype(result)
    p = (T(1), T(2))
    v0 = (T(3), T(4))
    v1 = (T(5), T(6))
    result[i] = JustPIC.second_stage(RungeKutta2(α), T(0.1), v0, v1, p)[1]
end

@testset "RK2 non-midpoint stage on backend" begin
    # Enough evaluations to expose per-particle allocations in the CUDA device heap.
    result = TA(backend)(zeros(FT, 2^21))
    JustPIC.launch!(backend(), rk2_stage_kernel!, length(result), result, FT(2 / 3))
    @test all(x -> x ≈ FT(1.45), Array(result))
end

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

@kernel function rk2_stages!(out, method, dt, p, v0, v1, backtracking)
    i = @index(Global, Linear)
    p1 = JustPIC.first_stage(method, dt, v0, p; backtracking = backtracking)
    p2 = JustPIC.second_stage(method, dt, v0, v1, p; backtracking = backtracking)
    for d in eachindex(p)
        out[d, i] = p1[d]
        out[length(p) + d, i] = p2[d]
    end
end

@testset "RK2 stages on backend" begin
    # Enough evaluations to expose per-thread heap allocations on CUDA Julia 1.10.
    n = 2^21
    for N in (2, 3), α in (0.5, 2 / 3), backtracking in (false, true)
        p = ntuple(d -> FT(d), N)
        v0 = ntuple(d -> FT(d + 2), N)
        v1 = ntuple(d -> FT(d + 2) + FT(0.1), N)
        dt = FT(0.1)
        method = RungeKutta2(FT(α))
        out = TA(backend)(zeros(FT, 2N, n))
        JustPIC.launch!(JustPIC.ka_backend(out), rk2_stages!, n, out, method, dt, p, v0, v1, backtracking)
        direction = backtracking ? -one(FT) : one(FT)
        a = FT(α)
        expected1 = p .+ direction .* a .* dt .* v0
        weight = inv(2a)
        expected2 = p .+ direction .* dt .* ((one(FT) - weight) .* v0 .+ weight .* v1)
        result = Array(out)
        @test eltype(result) === FT
        for d in 1:N
            @test all(x -> isapprox(x, expected1[d]), @view result[d, :])
            @test all(x -> isapprox(x, expected2[d]), @view result[N + d, :])
        end
    end
end

@testset "Euler" begin
    @test Euler() isa Euler
    @test Euler(1) isa Euler

    p1 = JustPIC.first_stage(Euler(), 0.1, (3.0, 4.0, 5.0), (1.0, 2.0, 3.0))
    @test p1 == (1.3, 2.4, 3.5)
    @test JustPIC.first_stage(Euler(), 0.1, (3.0, 4.0), (1.0, 2.0)) == p1[1:2]
end
