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
# a floating-point type that differs from the particle precision
const OTHER_FT = FT === Float64 ? Float32 : Float16

include(joinpath(@__DIR__, "helpers_backend.jl"))
check_backend(BACKEND_NAME, backend, FT)

const dev = TA(backend)

struct UnsupportedIntegrator <: AbstractAdvectionIntegrator end

function staggered_grid(n)
    xv = range(FT(0), FT(1); length = n + 1)
    dx = step(xv)
    xc_ghosted = range(dx / 2 - dx, 1 - dx / 2 + dx; length = n + 2)
    return (xv, xc_ghosted), (xc_ghosted, xv)
end

function test_particles(n = 6; nxcell = 16)
    grid_vx, grid_vy = staggered_grid(n)
    return init_particles(backend, nxcell, 32, 4, grid_vx, grid_vy)
end

n_active(particles) = count(Array(particles.index.data))
vertex_size(particles) = length.(particles.xvi)
center_size(particles) = length.(particles.xci)
velocity(particles, T = FT) = ntuple(d -> dev(zeros(T, length.(particles.xi_vel[d]))), 2)

@testset "init_particles argument checks" begin
    grid_vx, grid_vy = staggered_grid(6)
    xv, xc = grid_vx
    @test_throws "strictly increasing" init_particles(backend, 16, 32, 4, (reverse(collect(xv)), xc), grid_vy)
    @test_throws "finite" init_particles(backend, 16, 32, 4, (vcat(collect(xv[1:(end - 1)]), FT(Inf)), xc), grid_vy)
    @test_throws DimensionMismatch init_particles(backend, 16, 32, 4, (xv, xc[1:(end - 1)]), grid_vy)
    @test_throws "share one precision" init_particles(backend, 16, 32, 4, (OTHER_FT.(xv), xc), grid_vy)
    @test_throws "`min_xcell` must be non-negative" init_particles(backend, 16, 32, -1, grid_vx, grid_vy)
    @test_throws "`nxcell` must be non-negative" init_particles(backend, -1, 32, 4, grid_vx, grid_vy)
    @test JustPIC.supports_float64(CPU)
    if BACKEND_NAME == "Metal"
        @test_throws "does not support Float64" init_particles(backend, 16, 32, 4, map(x -> Float64.(x), grid_vx), map(x -> Float64.(x), grid_vy))
    end
end

@testset "move_particles! argument checks" begin
    particles = test_particles()
    pT, pP = init_cell_arrays(particles, Val(2))
    other, = init_cell_arrays(test_particles(4), Val(1))

    @test_throws "must be a tuple" move_particles!(particles, pT)
    @test_throws DimensionMismatch move_particles!(particles, (other,))
    @test_throws "floating point" move_particles!(particles, (particles.index,))
    @test_throws "share memory" move_particles!(particles, (pT, pT))
    @test_throws "share memory" move_particles!(particles, (particles.coords[1],))
    move_particles!(particles, (pT, pP))

    # NaN in an active slot is invalid; the container is left untouched
    nan_particles = test_particles()
    fill!(nan_particles.coords[1].data, FT(NaN))
    active = n_active(nan_particles)
    @test_throws "NaN coordinates" move_particles!(nan_particles, ())
    @test n_active(nan_particles) == active

    # infinite coordinates mark particles that left the domain; they are removed
    escaped = test_particles()
    fill!(escaped.coords[1].data, FT(Inf))
    move_particles!(escaped, ())
    @test n_active(escaped) == 0
end

@testset "advection! argument checks" begin
    particles = test_particles()
    V = velocity(particles)
    dt = FT(0.01)
    for advect! in (advection!, advection_LinP!, advection_MQS!)
        @test_throws "unsupported advection integrator" advect!(particles, UnsupportedIntegrator(), V, dt)
        @test_throws "`dt` must be finite" advect!(particles, RungeKutta2(), V, FT(NaN))
        @test_throws "tuple of 2 velocity components" advect!(particles, RungeKutta2(), V[1:1], dt)
        @test_throws DimensionMismatch advect!(particles, RungeKutta2(), (V[1], V[1]), dt)
        @test_throws "mixed precision" advect!(particles, RungeKutta2(), velocity(particles, OTHER_FT), dt)
    end
    advection!(particles, RungeKutta2(), V, dt)
end

@testset "interpolation argument checks" begin
    particles = test_particles()
    pT, pP = init_cell_arrays(particles, Val(2))
    F = dev(zeros(FT, vertex_size(particles)))
    G = dev(zeros(FT, vertex_size(particles)))

    @test_throws DimensionMismatch particle2grid!(F, pT, particles; ghost_1 = false)
    @test_throws DimensionMismatch grid2particle!(pT, F, particles; ghost_2 = false)
    particle2grid!(dev(zeros(FT, vertex_size(particles) .- (2, 0))), pT, particles; ghost_1 = false)
    @test_throws "single fields or tuples of the same length" particle2grid!((F, G), pT, particles)
    @test_throws "share memory" particle2grid!((F, F), (pT, pP), particles)
    @test_throws "mixed precision" particle2grid!(dev(zeros(OTHER_FT, vertex_size(particles))), pT, particles)
    @test_throws "CellArray" grid2particle!(dev(zeros(FT, 10)), F, particles)
    @test_throws "`α` must lie in [0, 1]" grid2particle_flip!(pT, F, G, particles; α = 2)
    @test_throws DimensionMismatch grid2particle_flip!(pT, F, dev(zeros(FT, 3, 3)), particles)

    C = dev(zeros(FT, center_size(particles)))
    @test_throws DimensionMismatch particle2centroid!(F, pT, particles)
    @test_throws DimensionMismatch centroid2particle!(pT, F, particles)
    @test_throws DimensionMismatch centroid2particle!(pT, C, particles; ghosted = false)
    particle2centroid!(C, pT, particles)
    centroid2particle!(pT, C, particles)
end

@testset "empty support is NaN" begin
    particles = test_particles(; nxcell = 0)
    @test n_active(particles) == 0
    pT, = init_cell_arrays(particles, Val(1))

    F = dev(zeros(FT, vertex_size(particles) .- 2))
    particle2grid!(F, pT, particles; ghost_1 = false, ghost_2 = false)
    @test all(isnan, Array(F))

    C = dev(zeros(FT, center_size(particles) .- 2))
    particle2centroid!(C, pT, particles; ghost_1 = false, ghost_2 = false)
    @test all(isnan, Array(C))

    phases, = init_cell_arrays(particles, Val(1))
    ratios = PhaseRatios(FT, backend, 2, size(particles.index) .- 2)
    update_phase_ratios!(ratios, particles, phases)
    for field in (ratios.center, ratios.vertex, ratios.Vx, ratios.Vy)
        @test all(isnan, Array(field.data))
    end
end

@testset "phase ratio argument checks" begin
    particles = test_particles()
    phases, = init_cell_arrays(particles, Val(1))
    ni = size(particles.index) .- 2
    ratios = PhaseRatios(FT, backend, 2, ni)

    fill!(phases.data, FT(1))
    update_phase_ratios!(ratios, particles, phases)
    data = Array(ratios.center.data)
    @test sum(data) ≈ prod(ni)
    @test all(x -> x ≈ 0 || x ≈ 1, data)

    for label in (FT(3), FT(0), FT(1.5))
        fill!(phases.data, label)
        @test_throws "phase label" update_phase_ratios!(ratios, particles, phases)
        @test_throws "phase label" phase_ratios_center!(ratios, particles, phases)
        @test_throws "phase label" phase_ratios_vertex!(ratios, particles, phases)
    end
    fill!(phases.data, FT(2))
    @test_throws DimensionMismatch update_phase_ratios!(PhaseRatios(FT, backend, 2, ni .+ 1), particles, phases)
    @test_throws "are 3D but the particles are 2D" update_phase_ratios!(PhaseRatios(FT, backend, 2, (ni..., 2)), particles, phases)
    @test_throws "only defined in 3D" phase_ratios_midpoint!(ratios.center, particles, phases, :xy)
    @test_throws "`nphases` must be at least 1" PhaseRatios(FT, backend, 0, ni)
    @test_throws "floating point" PhaseRatios(Int, backend, 2, ni)
end

@testset "injection argument checks" begin
    particles = test_particles()
    pT, = init_cell_arrays(particles, Val(1))
    phases, = init_cell_arrays(particles, Val(1))
    fill!(phases.data, FT(1))
    @test_throws "must be a tuple" inject_particles!(particles, pT)
    @test_throws "share memory" inject_particles!(particles, (pT, pT))
    @test_throws DimensionMismatch inject_particles_phase!(particles, phases, (pT,), (dev(zeros(FT, 2, 2)),))
    @test_throws "one grid field per entry" inject_particles_phase!(particles, phases, (pT,), ())
    inject_particles_phase!(particles, phases, (pT,), (dev(zeros(FT, size(particles.index) .- 2)),))
end

@testset "subgrid diffusion argument checks" begin
    particles = test_particles()
    pT, = init_cell_arrays(particles, Val(1))
    arrays = SubgridDiffusionCellArrays(particles)
    T = dev(zeros(FT, vertex_size(particles)))
    ΔT = dev(zeros(FT, size(pT)))
    @test_throws "`d` must lie in [0, 1]" subgrid_diffusion!(pT, T, ΔT, arrays, particles, FT(1); d = 2)
    @test_throws "`dt` must be finite" subgrid_diffusion!(pT, T, ΔT, arrays, particles, FT(Inf))
    @test_throws DimensionMismatch subgrid_diffusion!(pT, dev(zeros(FT, 3, 3)), ΔT, arrays, particles, FT(1))
end

@testset "passive marker checks" begin
    xv = range(FT(0), FT(1); length = 5)
    xvi = (xv, xv)
    xc = range(FT(0.125) - FT(0.25), FT(0.875) + FT(0.25); length = 6)
    grid_vxi = ((xv, xc), (xc, xv))

    @test_throws DimensionMismatch init_passive_markers(backend, (dev(FT[0.5, 0.5]), dev(FT[0.5])))
    @test_throws "finite" init_passive_markers(backend, (dev(FT[0.5, NaN]), dev(FT[0.5, 0.5])))
    markers = init_passive_markers(backend, (dev(FT[0.2, 0.9]), dev(FT[0.5, 0.1])))

    F = dev(ones(FT, length.(xvi)))
    buffer = similar(F)
    values = dev(zeros(FT, 2))
    grid2particle!(values, xvi, F, markers)
    @test Array(values) ≈ ones(FT, 2)
    @test_throws DimensionMismatch grid2particle!(dev(zeros(FT, 3)), xvi, F, markers)
    @test_throws DimensionMismatch grid2particle!(values, xvi, dev(ones(FT, 3, 3)), markers)
    @test_throws "share memory" particle2grid!(F, values, F, xvi, markers)
    @test_throws "single field" particle2grid!((F,), (values,), buffer, xvi, markers)
    @test_throws "unsupported advection integrator" advection!(markers, UnsupportedIntegrator(), (F, F), grid_vxi, FT(1))

    outside = init_passive_markers(backend, (dev(FT[0.2, 1.5]), dev(FT[0.5, 0.1])))
    @test_throws "outside the grid" grid2particle!(values, xvi, F, outside)
    @test_throws "outside the grid" particle2grid!(F, values, buffer, xvi, outside)

    # a flow that carries the markers across the boundary stops them on it
    V = ntuple(d -> dev(fill(FT(d == 1 ? 1 : -1), length.(grid_vxi[d]))), 2)
    advection!(markers, Euler(), V, grid_vxi, FT(2))
    @test Array(markers.coords[1]) == FT[1, 1]
    @test Array(markers.coords[2]) == FT[0, 0]
end

@testset "update_cell_halo! requires ImplicitGlobalGrid" begin
    particles = test_particles()
    @test_throws "requires an ImplicitGlobalGrid" update_cell_halo!(particles.coords...)
end
