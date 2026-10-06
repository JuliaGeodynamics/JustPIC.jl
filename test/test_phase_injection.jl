const BACKEND_NAME = get(ENV, "JULIA_JUSTPIC_BACKEND", "CPU")
@static if BACKEND_NAME == "Metal"
    using Metal
elseif BACKEND_NAME == "CUDA"
    using CUDA
elseif BACKEND_NAME == "AMDGPU"
    using AMDGPU
end

using JustPIC, Test, CellArrays
import KernelAbstractions: CPU
const backend = BACKEND_NAME == "CPU" ? CPU :
    BACKEND_NAME == "Metal" ? Metal.MetalBackend :
    BACKEND_NAME == "CUDA" ? CUDA.CUDABackend : AMDGPU.ROCBackend
const FT = BACKEND_NAME == "Metal" || get(ENV, "JULIA_JUSTPIC_PRECISION", "") == "Float32" ? Float32 : Float64
include("helpers_backend.jl")
check_backend(BACKEND_NAME, backend, FT)
BACKEND_NAME != "CPU" && getproperty(Main, Symbol(BACKEND_NAME)).allowscalar(false)

function mixed_injection_prototype(::Val{N}, refined) where {N}
    ni = ntuple(d -> 3 + d, Val(N))
    xv = ntuple(Val(N)) do d
        x = collect(range(zero(FT), one(FT); length = ni[d] + 1))
        refined ? x .^ 2 : x
    end
    xc = map(x -> (x[1:(end - 1)] .+ x[2:end]) ./ 2, xv)
    xcg = map(x -> vcat(2x[1] - x[2], x, 2x[end] - x[end - 1]), xc)
    grids = ntuple(d -> ntuple(k -> k == d ? xv[k] : xcg[k], Val(N)), Val(N))
    particles = init_particles(backend, 2^N, 2^(N + 1), 2^N, grids...)
    maxslots = particles.max_xcell
    # Keep one live phase donor per physical cell and force every other quadrant to refill.
    for ip in 2:maxslots
        fill!(field(particles.index, ip), false)
        foreach(A -> fill!(field(A, ip), FT(NaN)), particles.coords)
    end
    before = to_cpu(JustPIC._copy(particles.index))
    phases, = init_cell_arrays(particles, Val(1))
    fill!(phases.data, FT(2))
    p_empty = copy(particles)
    inject_particles_phase!(p_empty, phases, (), ())
    @test count(Array(p_empty.index.data)) > count(Array(particles.index.data))
    # All four layouts, plus halos on only some axes, in a single injection call.
    specs = ((true, false), (true, true), (false, false), (false, true), (true, :mixed), (false, :mixed))
    field_grids = map(specs) do (center, ghosts)
        base = map(to_cpu, center ? particles.xci : particles.xvi)
        ntuple(d -> ghosts === true || (ghosts === :mixed && isodd(d)) ? base[d] : base[d][2:(end - 1)], Val(N))
    end
    for axis in 1:N
        p = copy(particles)
        args = init_cell_arrays(p, Val(length(specs) + 1))
        foreach(A -> fill!(A.data, FT(-99)), args)
        fields = map(field_grids) do xi
            F = [FT(2) + FT(3) * xi[axis][I[axis]] for I in CartesianIndices(length.(xi))]
            TA(backend)(F)
        end
        # A view adds heterogeneous array types to the tuple, not just heterogeneous sizes.
        fields = (fields[1], view(fields[2], axes(fields[2])...), fields[3:end]..., TA(backend)(fill(FT(7), ni)))
        inject_particles_phase!(p, phases, args, fields)
        idx, coords, values = to_cpu(p.index), map(to_cpu, p.coords), map(to_cpu, args)
        phases_cpu = to_cpu(phases)
        added = 0
        boundary_added = 0
        for I in CartesianIndices(size(idx)), ip in 1:maxslots
            idx[I][ip] || continue
            if before[I][ip]
                @test all(A -> A[I][ip] == FT(-99), values)
                continue
            end
            added += 1
            boundary_added += any(d -> I[d] in (2, ni[d] + 1), 1:N)
            x = coords[axis][I][ip]
            @test phases_cpu[I][ip] == FT(2)
            for j in eachindex(specs)
                samples = field_grids[j][axis]
                expected = FT(2) + FT(3) * clamp(x, first(samples), last(samples))
                @test isapprox(values[j][I][ip], expected; atol = 32eps(FT), rtol = 32eps(FT))
            end
            @test values[end][I][ip] ≈ FT(7)
        end
        @test added > 0
        @test boundary_added > 0
        for I in CartesianIndices(ni)
            @test count(idx[Tuple(I) .+ 1...]) >= p.min_xcell
        end
        @test_throws DimensionMismatch inject_particles_phase!(p, phases, (args[1],), (TA(backend)(zeros(FT, ntuple(_ -> 2, Val(N)))),))
    end
    return
end

@testset "Mixed phase injection layouts" begin
    for N in (2, 3), refined in (false, true)
        @testset "$(N)D, refined=$refined" mixed_injection_prototype(Val(N), refined)
    end
end

# Staggered velocity grids with one ghost center per side; vertex coordinates `xv`.
function injection_grids(xv::NTuple{N}) where {N}
    xc = map(x -> (x[1:(end - 1)] .+ x[2:end]) ./ 2, xv)
    xcg = map(x -> vcat(2x[1] - x[2], x, 2x[end] - x[end - 1]), xc)
    return ntuple(d -> ntuple(k -> k == d ? xv[k] : xcg[k], Val(N)), Val(N))
end

# Keep only the first `nkeep` slots of every cell live.
function thin_particles!(particles, nkeep)
    for ip in (nkeep + 1):particles.max_xcell
        fill!(field(particles.index, ip), false)
    end
    return particles
end

# Brute-force nearest pre-existing particle in the 3^N stencil around cell I.
function nearest_donor_cpu(before, coords, I, x)
    best, found = typemax(FT), nothing
    stencil = CartesianIndices(ntuple(d -> max(I[d] - 1, 1):min(I[d] + 1, size(before, d)), length(x)))
    for J in stencil, ip in eachindex(before[J])
        before[J][ip] || continue
        dist = sqrt(sum(k -> (coords[k][J][ip] - x[k])^2, eachindex(x)))
        dist < best && ((best, found) = (dist, (J, ip)))
    end
    return found
end

function injection_donor_snapshot(::Val{N}) where {N}
    ni = ntuple(d -> 3 + d, Val(N))
    xv = ntuple(d -> collect(range(zero(FT), one(FT); length = ni[d] + 1)), Val(N))
    particles = thin_particles!(init_particles(backend, 2^N, 2^(N + 1), 2^N, injection_grids(xv)...), 1)
    before = to_cpu(JustPIC._copy(particles.index))
    coords0 = map(to_cpu, particles.coords)
    pid, = init_cell_arrays(particles, Val(1))
    for ip in 1:particles.max_xcell
        copyto!(field(pid, ip), TA(backend)([FT(n * particles.max_xcell + ip) for n in LinearIndices(size(pid))]))
    end
    pid_cpu = to_cpu(pid)
    inject_particles!(particles, (pid,))
    idx, coords, vals = to_cpu(particles.index), map(to_cpu, particles.coords), to_cpu(pid)
    added = 0
    for I in CartesianIndices(size(idx)), ip in 1:particles.max_xcell
        (idx[I][ip] && !before[I][ip]) || continue
        J, k = nearest_donor_cpu(before, coords0, I, ntuple(d -> coords[d][I][ip], Val(N)))
        @test vals[I][ip] == pid_cpu[J][k]
        added += 1
    end
    @test added > 0
    @test all(I -> count(idx[I]) ≥ particles.min_xcell, CartesianIndices(ntuple(d -> 2:(ni[d] + 1), Val(N))))
    return
end

@testset "Injection reads only the donor snapshot" begin
    for N in (2, 3)
        @testset "$(N)D" injection_donor_snapshot(Val(N))
    end
end

@testset "Injection scheme keyword" begin
    xv = collect(range(zero(FT), one(FT); length = 5))
    particles = thin_particles!(init_particles(backend, 4, 8, 4, injection_grids((xv, xv))...), 1)
    pT, phases = init_cell_arrays(particles, Val(2))
    T = TA(backend)([FT(1) + FT(2) * x for x in xv, y in xv])
    @test_throws "unknown injection scheme" inject_particles!(particles, (pT,), (T,); scheme = :foo)
    @test_throws "unknown injection scheme" inject_particles_phase!(particles, phases, (pT,), (T,); scheme = :foo)
    n0 = count(Array(particles.index.data))
    inject_particles!(particles, (pT,), (T,))
    @test count(Array(particles.index.data)) > n0
    idx, px, vals = to_cpu(particles.index), to_cpu(particles.coords[1]), to_cpu(pT)
    for I in CartesianIndices(size(idx)), ip in 2:particles.max_xcell
        idx[I][ip] && @test vals[I][ip] ≈ FT(1) + FT(2) * px[I][ip] atol = 32eps(FT)
    end
end

# Every injected value equals the linear field clamped to the range carried by donors in
# the 3^N neighborhood; the grid fallback is poisoned so that any fallback shows.
function least_squares_injection(::Val{N}, refined) where {N}
    ni = ntuple(d -> 3 + d, Val(N))
    xv = ntuple(Val(N)) do d
        x = collect(range(zero(FT), one(FT); length = ni[d] + 1))
        refined ? x .^ 2 : x
    end
    # N + 2 donors per cell: enough for a cell-local fit, too few to fill every quadrant.
    particles = thin_particles!(init_particles(backend, 2^(N + 1), 2^(N + 2), 2^(N + 1), injection_grids(xv)...), N + 2)
    F(x) = FT(1) + sum(d -> FT(d + 1) * x[d], 1:N)
    before = to_cpu(JustPIC._copy(particles.index))
    pF, = init_cell_arrays(particles, Val(1))
    for ip in 1:particles.max_xcell
        x = map(c -> Array(field(c, ip)), particles.coords)
        copyto!(field(pF, ip), TA(backend)([F(ntuple(d -> x[d][I], Val(N))) for I in CartesianIndices(x[1])]))
    end
    pF_cpu = to_cpu(pF)
    poison = TA(backend)(fill(FT(-1000), ni .+ 1))
    inject_particles!(particles, (pF,), (poison,); scheme = :least_squares)
    idx, coords, vals = to_cpu(particles.index), map(to_cpu, particles.coords), to_cpu(pF)
    added = 0
    for I in CartesianIndices(size(idx)), ip in 1:particles.max_xcell
        (idx[I][ip] && !before[I][ip]) || continue
        lo, hi = typemax(FT), typemin(FT)
        for J in CartesianIndices(ntuple(d -> max(I[d] - 1, 1):min(I[d] + 1, size(idx, d)), Val(N))), k in 1:particles.max_xcell
            before[J][k] || continue
            lo, hi = min(lo, pF_cpu[J][k]), max(hi, pF_cpu[J][k])
        end
        @test vals[I][ip] ≈ clamp(F(ntuple(d -> coords[d][I][ip], Val(N))), lo, hi) rtol = cbrt(eps(FT))
        added += 1
    end
    @test added > 0
    return
end

@testset "Least-squares injection" begin
    for N in (2, 3), refined in (false, true)
        @testset "$(N)D, refined=$refined" least_squares_injection(Val(N), refined)
    end
end

# Collinear donors (all on one line along x): the fit is degenerate in every cell, so
# values come from the grid field.
function least_squares_fallback(::Val{N}) where {N}
    xv = ntuple(_ -> collect(range(zero(FT), one(FT); length = 4)), Val(N))
    particles = thin_particles!(init_particles(backend, 2^N, 2^(N + 1), 2^N, injection_grids(xv)...), 1)
    for d in 2:N
        copyto!(field(particles.coords[d], 1), TA(backend)(fill(FT(0.5), size(field(particles.coords[d], 1)))))
    end
    before = to_cpu(JustPIC._copy(particles.index))
    pF, = init_cell_arrays(particles, Val(1))
    fill!(pF.data, FT(3))
    inject_particles!(particles, (pF,), (TA(backend)(fill(FT(7), ntuple(_ -> 4, Val(N)))),); scheme = :least_squares)
    idx, vals = to_cpu(particles.index), to_cpu(pF)
    added = 0
    for I in CartesianIndices(size(idx)), ip in 1:particles.max_xcell
        (idx[I][ip] && !before[I][ip]) || continue
        @test vals[I][ip] == FT(7)
        added += 1
    end
    @test added > 0
    return
end

@testset "Least-squares falls back to grid" begin
    for N in (2, 3)
        @testset "$(N)D" least_squares_fallback(Val(N))
    end
end

# A step field: every injected value stays inside the donor range of its 3^N neighborhood.
function least_squares_limiter(::Val{N}) where {N}
    xv = ntuple(_ -> collect(range(zero(FT), one(FT); length = 7)), Val(N))
    particles = thin_particles!(init_particles(backend, 2^(N + 1), 2^(N + 2), 2^(N + 1), injection_grids(xv)...), N + 2)
    before = to_cpu(JustPIC._copy(particles.index))
    pF, = init_cell_arrays(particles, Val(1))
    for ip in 1:particles.max_xcell
        x = Array(field(particles.coords[1], ip))
        copyto!(field(pF, ip), TA(backend)(FT.(x .> FT(0.5))))
    end
    pF_cpu = to_cpu(pF)
    inject_particles!(particles, (pF,), (TA(backend)(fill(FT(-1000), ntuple(_ -> 7, Val(N)))),); scheme = :least_squares)
    idx, vals = to_cpu(particles.index), to_cpu(pF)
    clamped = 0
    for I in CartesianIndices(size(idx)), ip in 1:particles.max_xcell
        (idx[I][ip] && !before[I][ip]) || continue
        stencil = CartesianIndices(ntuple(d -> max(I[d] - 1, 1):min(I[d] + 1, size(idx, d)), Val(N)))
        donors = [pF_cpu[J][k] for J in stencil for k in 1:particles.max_xcell if before[J][k]]
        @test minimum(donors) ≤ vals[I][ip] ≤ maximum(donors)
        clamped += vals[I][ip] in (minimum(donors), maximum(donors)) && minimum(donors) < maximum(donors)
    end
    @test clamped > 0
    return
end

@testset "Least-squares limiter" begin
    for N in (2, 3)
        @testset "$(N)D" least_squares_limiter(Val(N))
    end
end

# Grid-based schemes need no donor without phases: a cell with an empty neighborhood is refilled.
function injection_into_void(::Val{N}, scheme) where {N}
    xv = ntuple(_ -> collect(range(zero(FT), one(FT); length = 8)), Val(N))
    particles = init_particles(backend, 2^N, 2^(N + 1), 2^N, injection_grids(xv)...)
    # Inner cells 3:7 (ghost-padded indices) hold no particle at all.
    for ip in 1:particles.max_xcell
        mask = Array(field(particles.index, ip))
        mask[ntuple(_ -> 3:7, Val(N))...] .= false
        copyto!(field(particles.index, ip), TA(backend)(mask))
    end
    pT, = init_cell_arrays(particles, Val(1))
    inject_particles!(particles, (pT,), (TA(backend)(fill(FT(5), ntuple(_ -> 8, Val(N)))),); scheme)
    idx, vals = to_cpu(particles.index), to_cpu(pT)
    center = ntuple(_ -> 5, Val(N))
    @test count(idx[center...]) ≥ particles.min_xcell
    @test all(ip -> !idx[center...][ip] || vals[center...][ip] == FT(5), 1:particles.max_xcell)
    return
end

@testset "Grid schemes refill voids without phases" begin
    for N in (2, 3), scheme in (:grid, :least_squares)
        @testset "$(N)D $scheme" injection_into_void(Val(N), scheme)
    end
end

# Normal equations from deterministic donor offsets ξ ∈ [-1/2, 1/2)^3 (quadratic Weyl
# sequence, generated in Float64), optionally forced onto the plane z = x + y + 0.37.
function fit_system(set, n, coplanar)
    α = (0.6180339887, 0.4142135624, 0.7320508076)
    M = zero(JustPIC.SMatrix{4, 4, FT})
    for k in (n * set + 1):(n * set + n)
        x, y, z = map(a -> FT(mod(k * k * a, 1.0) - 0.5), α)
        φ = JustPIC.SVector(one(FT), x, y, coplanar ? x + y + FT(0.37) : z)
        M += φ * φ'
    end
    return (M, (zero(JustPIC.SVector{4, FT}),), n)
end

@testset "Least-squares degeneracy threshold" begin
    @test count(set -> first(JustPIC.solve_fit(fit_system(set, 4, false))), 1:2000) ≥ 0.7 * 2000
    @test !any(set -> first(JustPIC.solve_fit(fit_system(set, 8, true))), 1:2000)
end

@testset "Injection input checks" begin
    xv = collect(range(zero(FT), one(FT); length = 5))
    particles = thin_particles!(init_particles(backend, 4, 8, 4, injection_grids((xv, xv))...), 1)
    pT, = init_cell_arrays(particles, Val(1))
    T = TA(backend)(zeros(FT, 5, 5))
    @test_throws "unknown injection scheme" inject_particles!(particles, (pT,), (T,); scheme = "grid")
    if BACKEND_NAME != "Metal"
        other = FT === Float64 ? Float32 : Float64
        p_other = init_particles(backend, 4, 8, 4, map(g -> map(x -> other.(x), g), injection_grids((xv, xv)))...)
        pX, = init_cell_arrays(p_other, Val(1))
        @test_throws "mixed precision is not supported" inject_particles!(particles, (pX,), (T,); scheme = :least_squares)
    end
end
