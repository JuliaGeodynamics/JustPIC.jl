const BACKEND_NAME = get(ENV, "JULIA_JUSTPIC_BACKEND", "CPU")
@static if BACKEND_NAME == "CUDA"
    using CUDA
elseif BACKEND_NAME == "AMDGPU"
    using AMDGPU
elseif BACKEND_NAME == "Metal"
    using Metal
end
using JustPIC, Test
import CellArraysIndexing as CAI
import CellArrays: blocklength
import KernelAbstractions: CPU
const backend = BACKEND_NAME == "CUDA" ? CUDA.CUDABackend :
    BACKEND_NAME == "AMDGPU" ? AMDGPU.ROCBackend :
    BACKEND_NAME == "Metal" ? Metal.MetalBackend : CPU
const FT = BACKEND_NAME == "Metal" || get(ENV, "JULIA_JUSTPIC_PRECISION", "") == "Float32" ? Float32 : Float64
include("helpers_backend.jl")
check_backend(BACKEND_NAME, backend, FT)

# Device CellArrays store all cells in one struct-of-arrays block, unlike the
# block-length-1 host copies returned by `to_cpu`.
upload!(dst, src) = copyto!(dst.data, isone(blocklength(dst)) ? src.data : permutedims(src.data, (3, 2, 1)))

function surface_fixture(; refined = false, capacity = 100, initial = 60, sloping = false)
    xv = refined ? FT[0, 0.8, 1.4, 3] : FT[0, 1, 2, 3]
    yv = FT[0, 1, 2]
    xc, yc = (xv[1:(end - 1)] .+ xv[2:end]) ./ 2, FT[0.5, 1.5]
    ghostxc = vcat(xc[1] - (xv[2] - xv[1]), xc, xc[end] + (xv[end] - xv[end - 1]))
    p = init_particles(backend, initial, capacity, 50, (xv, FT[-0.5, 0.5, 1.5, 2.5]), (ghostxc, yv))
    phases, history, temperature = init_cell_arrays(p, Val(3))
    fill!(phases.data, FT(5))
    fill!(history.data, FT(-100))
    fill!(temperature.data, FT(0))
    h = sloping ? FT[-0.01, 0.02, -0.01, 0.02] : FT[0.001, 0.003, 0.002, 0.004]
    chain = init_markerchain(backend, 4, 2, 8, xv, h)
    # One surviving rock donor in the middle column must support both inflow edges.
    coords = JustPIC._copy.(p.coords)
    hostcoords = to_cpu.(coords)
    hostphases, hosthistory, hosttemperature = to_cpu.((JustPIC._copy(phases), JustPIC._copy(history), JustPIC._copy(temperature)))
    CAI.@index hostcoords[1][1, 3, 2] = xc[2]
    CAI.@index hostcoords[2][1, 3, 2] = FT(0.001)
    CAI.@index hostphases[1, 3, 2] = FT(1)
    CAI.@index hosthistory[1, 3, 2] = FT(123)
    CAI.@index hosttemperature[1, 3, 2] = FT(800)
    foreach(upload!, p.coords, hostcoords)
    foreach(upload!, (phases, history, temperature), (hostphases, hosthistory, hosttemperature))
    return p, phases, (history, temperature), chain, xv, yv
end

function check_surface_support(p, phases, args, chain, xv, yv)
    index, phase = to_cpu.((p.index, phases))
    coords, fields = to_cpu.(p.coords), to_cpu.(args)
    h = Array(chain.h_vertices)
    for j in 2:(size(index, 2) - 1), i in 2:(size(index, 1) - 1)
        x0, x1 = xv[i - 1], xv[i]
        y0, y1 = yv[j - 1], yv[j]
        for ip in cellaxes(index)
            (CAI.@index index[ip, i, j]) || continue
            x, y = (CAI.@index coords[1][ip, i, j]), (CAI.@index coords[2][ip, i, j])
            ph = CAI.@index phase[ip, i, j]
            surface = h[i - 1] + (h[i] - h[i - 1]) * (x - x0) / (x1 - x0)
            @test x0 ≤ x ≤ x1 && y0 ≤ y ≤ y1
            @test (y > surface) == (ph == 5)
            @test all(A -> isfinite(CAI.@index A[ip, i, j]), fields)
        end
        if j == 2
            for right in (false, true)
                @test any(cellaxes(index)) do ip
                    (CAI.@index index[ip, i, j]) || return false
                    x = CAI.@index coords[1][ip, i, j]
                    ph = CAI.@index phase[ip, i, j]
                    return ph != 5 && ((x ≥ (x0 + x1) / 2) == right)
                end
            end
        end
    end
    return @test all(!any(index[i, j]) for i in axes(index, 1), j in axes(index, 2) if i in (1, size(index, 1)) || j in (1, size(index, 2)))
end

@testset "Surface-aware injection" begin
    for refined in (false, true), sloping in (false, true)
        p, phases, args, chain, xv, yv = surface_fixture(; refined, sloping)
        saved = copy.(getproperty.(to_cpu.((p.coords..., phases, args...)), :data))
        original_index = to_cpu(JustPIC._copy(p.index))
        original_coords = to_cpu.(JustPIC._copy.(p.coords))
        original_phases = to_cpu(JustPIC._copy(phases))
        heights = Array(chain.h_vertices)
        for j in 2:3, i in 2:4, ip in cellaxes(original_index)
            x = CAI.@index original_coords[1][ip, i, j]
            y = CAI.@index original_coords[2][ip, i, j]
            surface = heights[i - 1] + (heights[i] - heights[i - 1]) * (x - xv[i - 1]) / (xv[i] - xv[i - 1])
            phase = CAI.@index original_phases[ip, i, j]
            if (y > surface) != (phase == 5)
                CAI.@index original_index[ip, i, j] = false
            end
        end
        inject_particles_phase!(p, phases, args, chain; air_phase = 5)
        check_surface_support(p, phases, args, chain, xv, yv)
        # Valid pre-existing slots retain their values exactly, including stress/strain history.
        live = Array(original_index.data)
        @test all(to_cpu(p.index).data[live])
        for (A, before) in zip((p.coords..., phases, args...), saved)
            @test to_cpu(A).data[live] == before[live]
        end
        count_before = sum(Array(p.index.data))
        inject_particles_phase!(p, phases, args, chain; air_phase = 5)
        @test sum(Array(p.index.data)) == count_before
        phase, history, temperature = to_cpu.((phases, args...))
        index = to_cpu(p.index)
        for J in CartesianIndices(index), ip in cellaxes(index)
            (CAI.@index index[ip, Tuple(J)...]) || continue
            rock = (CAI.@index phase[ip, Tuple(J)...]) != 5
            @test (CAI.@index history[ip, Tuple(J)...]) == (rock ? FT(123) : FT(-100))
            @test (CAI.@index temperature[ip, Tuple(J)...]) == (rock ? FT(800) : FT(0))
        end
    end

    p, phases, args, chain, xv, yv = surface_fixture()
    # Remove the only rock donor: occupancy by air must not silently count as rock support.
    fill!(phases.data, FT(5))
    @test_throws "targets unmet" inject_particles_phase!(p, phases, args, chain; air_phase = 5)

    p, phases, args, chain, xv, yv = surface_fixture()
    initializer = (position, is_air) -> (is_air ? FT(5) : FT(2), (is_air ? FT(-7) : FT(42), is_air ? FT(0) : FT(900)))
    inject_particles_phase!(p, phases, args, chain; air_phase = 5, boundary_initializer = initializer)
    check_surface_support(p, phases, args, chain, xv, yv)
    phase, history = to_cpu.((phases, args[1]))
    @test any(==(FT(2)), phase[2, 2])
    @test any(==(FT(2)), phase[4, 2])
    @test all(v -> v in (FT(123), FT(-100), FT(42), FT(-7)), vcat((collect(history[i, j]) for i in 2:4, j in 2:3)...))

    p, phases, args, chain, xv, yv = surface_fixture()
    invalid_initializer = (position, is_air) -> (FT(5), (FT(NaN), FT(0)))
    @test_throws "targets unmet" inject_particles_phase!(
        p, phases, args, chain; air_phase = 5, boundary_initializer = invalid_initializer,
    )

    p, phases, args, chain, xv, yv = surface_fixture(; capacity = 60)
    # A full, valid all-air edge cell has no free slot for its required rock support.
    px, py = to_cpu.(JustPIC._copy.(p.coords))
    for ip in cellaxes(p.index)
        CAI.@index px[ip, 2, 2] = FT(0.25) + FT(isodd(ip)) / 2
        CAI.@index py[ip, 2, 2] = FT(0.25) + FT((ip % 4) ≥ 2) / 2
    end
    upload!(p.coords[1], px)
    upload!(p.coords[2], py)
    @test_throws "targets unmet" inject_particles_phase!(p, phases, args, chain; air_phase = 5)
    @test_throws ArgumentError inject_particles_phase!(p, phases, args, chain; air_phase = 0)
    badchain = init_markerchain(backend, 4, 2, 8, FT[0, 1, 2, 4], FT(0))
    @test_throws DimensionMismatch inject_particles_phase!(p, phases, args, badchain; air_phase = 5)
end

@testset "Repeated extension and reversal" begin
    p, phases, args, chain, xv, yv = surface_fixture(; refined = true, capacity = 160)
    inject_particles_phase!(p, phases, args, chain; air_phase = 5)
    for step in 1:200
        rate = step ≤ 60 ? FT(0.005) : FT(-0.005)
        @. p.coords[1].data += rate * (p.coords[1].data - $(FT(1.5)))
        move_particles!(p, (phases, args...))
        inject_particles_phase!(p, phases, args, chain; air_phase = 5)
        check_surface_support(p, phases, args, chain, xv, yv)
        ratios = PhaseRatios(FT, backend, 5, (3, 2))
        update_phase_ratios!(ratios, p, phases)
        weights = to_cpu(ratios.center)
        @test all(w -> all(isfinite, w) && sum(w) ≈ 1, weights)
        @test all(i -> weights[i, 1][1] > 0, 1:3)
    end
end
