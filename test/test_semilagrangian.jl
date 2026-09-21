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

# Metal has no Float64; JULIA_JUSTPIC_PRECISION=Float32 runs the same paths on CPU
const FT = if BACKEND_NAME == "Metal" || get(ENV, "JULIA_JUSTPIC_PRECISION", "") == "Float32"
    Float32
else
    Float64
end

include(joinpath(@__DIR__, "helpers_backend.jl"))
check_backend(BACKEND_NAME, backend, FT)

sml_expand(x::AbstractVector) = vcat(x[1] - (x[2] - x[1]), x, x[end] + (x[end] - x[end - 1]))

# Refined grid, constant velocity and small enough displacement (< min dx) that every
# interior node backtracks into a neighbouring cell of the domain.
function check_semilagrangian(advect!, method, ::Val{N}) where {N}
    dev = TA(backend)
    xv = FT[0, 0.15, 0.35, 0.6, 0.8, 1]
    n = length(xv)
    xc = (xv[1:(end - 1)] .+ xv[2:end]) ./ 2
    xvi = ntuple(_ -> xv, Val(N))
    grid = map(dev, xvi)
    grid_vi = ntuple(d -> ntuple(j -> dev(j == d ? xv : sml_expand(xc)), Val(N)), Val(N))
    vsize(d) = length.(grid_vi[d])

    v = ntuple(d -> FT((0.06, -0.05, 0.04)[d]), Val(N))
    dt = FT(1)
    V0 = ntuple(d -> dev(zeros(FT, vsize(d))), Val(N))
    V = ntuple(d -> dev(fill(v[d], vsize(d))), Val(N))

    inner = ntuple(_ -> 2:(n - 1), Val(N))
    inner_xvi = map(x -> x[2:(end - 1)], xvi)
    field(f) = [f(p) for p in Iterators.product(xvi...)]
    translated(f) = [f(p .- v .* dt) for p in Iterators.product(inner_xvi...)]
    empty_field() = dev(fill(FT(-7), ntuple(_ -> n, Val(N))))

    ca = ntuple(d -> FT(d), Val(N))
    cb = ntuple(d -> FT(-1 - d), Val(N))
    fa = p -> FT(1) + sum(ca .* p)
    fb = p -> FT(5) + sum(cb .* p)
    tol = FT === Float32 ? 1.0f-5 : 1.0e-12

    # zero flow: destination reproduces the source, source and boundary nodes untouched
    source = field(p -> sin(FT(3) * sum(p)) + prod(p))
    F0 = dev(source)
    F = empty_field()
    advect!(F, F0, method, V0, grid_vi, grid, dt)
    Fh = Array(F)
    boundary = trues(size(source))
    boundary[inner...] .= false
    @test Fh[inner...] ≈ source[inner...] atol = tol rtol = tol
    @test all(==(FT(-7)), Fh[boundary])
    @test Array(F0) == source

    # translated affine fields, scalar and tuple variants agree
    source_a, source_b = field(fa), field(fb)
    F0a, F0b = dev(source_a), dev(source_b)
    Fs, Fa, Fb = empty_field(), empty_field(), empty_field()
    advect!(Fs, F0a, method, V, grid_vi, grid, dt)
    advect!((Fa, Fb), (F0a, F0b), method, V, grid_vi, grid, dt)
    @test Array(Fs)[inner...] ≈ translated(fa) atol = tol rtol = tol
    @test Array(Fa)[inner...] ≈ translated(fa) atol = tol rtol = tol
    @test Array(Fb)[inner...] ≈ translated(fb) atol = tol rtol = tol
    @test Array(Fs) ≈ Array(Fa) atol = tol rtol = tol
    @test Array(F0a) == source_a
    @test Array(F0b) == source_b

    # source and destination must not share memory
    @test_throws ArgumentError advect!(F0a, F0a, method, V, grid_vi, grid, dt)
    @test_throws ArgumentError advect!((F0a,), (F0a,), method, V, grid_vi, grid, dt)
    @test_throws ArgumentError advect!((Fa, F0b), (F0a, F0b), method, V, grid_vi, grid, dt)
    @test_throws ArgumentError advect!((F0b, Fa), (F0a, F0b), method, V, grid_vi, grid, dt)
    @test Array(F0a) == source_a
    @test Array(F0b) == source_b
    return nothing
end

const SML_SCHEMES = (
    "plain" => semilagrangian_advection!,
    "LinP" => semilagrangian_advection_LinP!,
    "MQS" => semilagrangian_advection_MQS!,
)

const SML_METHODS = (RungeKutta2(), RungeKutta4())

@testset "Semi-Lagrangian $(scheme) $(nameof(typeof(method))) $(N)D" for N in (2, 3),
        (scheme, advect!) in SML_SCHEMES,
        method in SML_METHODS

    check_semilagrangian(advect!, method, Val(N))
end
