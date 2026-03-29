@static if ENV["JULIA_JUSTPIC_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTPIC_BACKEND"] === "CUDA"
    using CUDA
end

using JLD2, JustPIC, Test
import JustPIC._2D as JP2
import JustPIC._3D as JP3

const backend = JustPIC.CPUBackend


function expand_range(x::AbstractRange)
    dx = x[2] - x[1]
    n = length(x)
    x1, x2 = extrema(x)
    xI = x1 - dx
    xF = x2 + dx
    return LinRange(xI, xF, n + 2)
end

function expand_range(x::AbstractVector)
    dx_left = x[2] - x[1]
    dx_right = x[end] - x[end - 1]
    x1, x2 = extrema(x)
    xI = x1 - dx_left
    xF = x2 + dx_right
    return vcat(xI, x, xF)
end

@testset "Save and load 2D" begin
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 6, 6, 6
    n = 64
    nx = ny = n - 1
    ni = nx, ny
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = LinRange(0, Lx, n), LinRange(0, Ly, n)
    dxi = dx, dy = xv[2] - xv[1], yv[2] - yv[1]
    # nodal centers
    xc, yc = LinRange(0 + dx / 2, Lx - dx / 2, n - 1), LinRange(0 + dy / 2, Ly - dy / 2, n - 1)
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc)
    grid_vy = expand_range(xc), yv
    grid_vel = grid_vx, grid_vx

    particles = JP2.init_particles(backend, nxcell, max_xcell, min_xcell, grid_vel...)
    phases, pT = JP2.init_cell_arrays(particles, Val(2))
    particle_args = (phases, pT)
    particle_args_reduced = (phases,)
    particle_args_kwarg = (phases,)
    phase_ratios = JP2.PhaseRatios(backend, 2, ni)
    initial_elevation = Ly / 2
    chain = JP2.init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, initial_elevation)
    @views particles.index.data[:, 1:3, 1] .= 1.0
    @views particles.index.data[:, 4:6, 1] .= 0.0

    JP2.checkpointing_particles(pwd(), particles; phases = phases, phase_ratios = phase_ratios, chain = chain, particle_args = particle_args, particle_args_reduced = particle_args_reduced, particle_args_kwarg = particle_args_kwarg)

    # test type conversion
    @test eltype(eltype(Array(phases))) === Float64
    @test eltype(eltype(Array(Float64, phases))) === Float64
    @test eltype(eltype(Array(Float32, phases))) === Float32
    @test eltype(eltype(Array(particles).coords[1].data)) === Float64
    @test eltype(eltype(Array(Float64, particles).coords[1].data)) === Float64
    @test eltype(eltype(Array(Float32, particles).coords[1].data)) === Float32
    @test eltype(eltype(Array(particles).index.data)) === Bool
    @test eltype(eltype(Array(Float32, particles).index.data)) === Bool
    @test eltype(eltype(Array(Float64, particles).index.data)) === Bool
    @test eltype(eltype(Array(phase_ratios).vertex.data)) === Float64
    @test eltype(eltype(Array(Float64, phase_ratios).vertex.data)) === Float64
    @test eltype(eltype(Array(Float32, phase_ratios).vertex.data)) === Float32

    jldsave(
        joinpath(pwd(), "particles.jld2");
        particles = Array(particles),
        phases = Array(phases),
        phase_ratios = Array(phase_ratios)
    )

    data = load(joinpath(pwd(), "particles.jld2"))
    particles2 = data["particles"]
    phases2 = data["phases"]
    phase_ratios2 = data["phase_ratios"]

    @test Array(particles).coords[1].data == particles2.coords[1].data
    @test Array(particles).coords[2].data == particles2.coords[2].data
    @test Array(particles).index.data == particles2.index.data
    @test Array(phase_ratios).center.data == phase_ratios2.center.data
    @test Array(phase_ratios).vertex.data == phase_ratios2.vertex.data
    @test Array(phases).data == phases2.data
    @test size(Array(particles).coords[1].data) == size(particles2.coords[1].data)
    @test size(Array(particles).coords[2].data) == size(particles2.coords[2].data)
    @test size(Array(particles).index.data) == size(particles2.index.data)
    @test size(Array(phase_ratios).center.data) == size(phase_ratios2.center.data)
    @test size(Array(phase_ratios).vertex.data) == size(phase_ratios2.vertex.data)
    @test size(Array(phases).data) == size(phases2.data)

    data1 = load(joinpath(pwd(), "particles_checkpoint.jld2"))
    particles3 = data1["particles"]
    phases3 = data1["phases"]
    phase_ratios3 = data1["phase_ratios"]
    chain3 = data1["chain"]
    particle_args3 = data1["particle_args"]
    particle_args_reduced3 = data1["particle_args_reduced"]
    particle_args_kwarg3 = data1["particle_args_kwarg"]

    @test chain3 isa JustPIC.MarkerChain{JustPIC.CPUBackend}
    @test particle_args3 isa Tuple
    @test particle_args_reduced3 isa Tuple
    @test particle_args_kwarg3 isa Tuple
    @test Array(particles).coords[1].data == particles3.coords[1].data
    @test Array(particles).coords[2].data == particles3.coords[2].data
    @test Array(particles).index.data == particles3.index.data
    @test Array(phase_ratios).center.data == phase_ratios3.center.data
    @test Array(phase_ratios).vertex.data == phase_ratios3.vertex.data
    @test Array(phases).data == phases3.data
    @test size(Array(particles).coords[1].data) == size(particles3.coords[1].data)
    @test size(Array(particles).coords[2].data) == size(particles3.coords[2].data)
    @test size(Array(particles).index.data) == size(particles3.index.data)
    @test size(Array(phase_ratios).center.data) == size(phase_ratios3.center.data)
    @test size(Array(phase_ratios).vertex.data) == size(phase_ratios3.vertex.data)
    @test size(Array(phases).data) == size(phases3.data)

    # Test on GPU card, if available
    isCUDA = isdefined(Main, :CUDA)
    isAMDGPU = isdefined(Main, :AMDGPU)

    if isCUDA || isAMDGPU
        T = isCUDA ? CuArray : ROCArray
        Backend = isCUDA ? CUDABackend : AMDGPUBackend

        particles2 = Array(particles)
        phases2 = Array(phases)
        phase_ratios2 = Array(phase_ratios)
        particles_gpu = T(particles2)
        phase_ratios_gpu = T(phase_ratios2)
        phases_gpu = T(phases2)
        particles_gpu2 = T(particles3)
        phase_ratios_gpu2 = T(phase_ratios3)
        phases_gpu2 = T(phases3)
        chain_gpu = T(chain)
        particle_args_gpu2 = T.(particle_args)
        particle_args_reduced_gpu2 = T.(particle_args_reduced)
        particle_args_kwarg_gpu2 = T.(particle_args_kwarg)

        @test particles_gpu isa JustPIC.Particles{Backend}
        @test phase_ratios_gpu isa JustPIC.PhaseRatios{Backend}
        @test last(typeof(phases_gpu).parameters) <: T{Float64, 3}
        @test size(particles_gpu.coords[1].data) == size(permutedims(particles.coords[1].data, (3, 2, 1)))
        @test size(particles_gpu.coords[2].data) == size(permutedims(particles.coords[2].data, (3, 2, 1)))
        @test size(particles_gpu.index.data) == size(permutedims(particles.index.data, (3, 2, 1)))
        @test size(phase_ratios_gpu.center.data) == size(permutedims(phase_ratios.center.data, (3, 2, 1)))
        @test size(phase_ratios_gpu.vertex.data) == size(permutedims(phase_ratios.vertex.data, (3, 2, 1)))
        @test size(phases_gpu.data) == size(permutedims(phases.data, (3, 2, 1)))

        @test particles_gpu2 isa JustPIC.Particles{Backend}
        @test phase_ratios_gpu2 isa JustPIC.PhaseRatios{Backend}
        @test chain_gpu isa JustPIC.MarkerChain{Backend}
        @test particle_args_gpu2 isa Tuple
        @test particle_args_reduced_gpu2 isa Tuple
        @test particle_args_kwarg_gpu2 isa Tuple
        @test last(typeof(phases_gpu2).parameters) <: T{Float64, 3}
        @test size(particles_gpu2.coords[1].data) == size(permutedims(particles.coords[1].data, (3, 2, 1)))
        @test size(particles_gpu2.coords[2].data) == size(permutedims(particles.coords[2].data, (3, 2, 1)))
        @test size(particles_gpu2.index.data) == size(permutedims(particles.index.data, (3, 2, 1)))
        @test size(phase_ratios_gpu2.center.data) == size(permutedims(phase_ratios.center.data, (3, 2, 1)))
        @test size(phase_ratios_gpu2.vertex.data) == size(permutedims(phase_ratios.vertex.data, (3, 2, 1)))
        @test size(phases_gpu2.data) == size(permutedims(phases.data, (3, 2, 1)))

        # test type conversion
        @test eltype(eltype(T(phases))) === Float64
        @test eltype(eltype(T(Float64, phases))) === Float64
        @test eltype(eltype(T(Float32, phases))) === Float32
        @test eltype(eltype(T(particles).coords[1].data)) === Float64
        @test eltype(eltype(T(Float64, particles).coords[1].data)) === Float64
        @test eltype(eltype(T(Float32, particles).coords[1].data)) === Float32
        @test eltype(eltype(T(phase_ratios).vertex.data)) === Float64
        @test eltype(eltype(T(Float64, phase_ratios).vertex.data)) === Float64
        @test eltype(eltype(T(Float32, phase_ratios).vertex.data)) === Float32
        @test eltype(eltype(T(particles).index.data)) === Bool
        @test eltype(eltype(T(Float32, particles).index.data)) === Bool
        @test eltype(eltype(T(Float64, particles).index.data)) === Bool
    end

    rm("particles_checkpoint.jld2") # cleanup
    rm("particles.jld2") # cleanup
end
@testset "Save and load 3D" begin
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 6, 6, 6
    n = 64
    nx = ny = nz = n - 1
    ni = nx, ny, nz
    Lx = Ly = Lz = 1.0
    Li = Lx, Ly, Lz
    # nodal vertices
    xvi = xv, yv, zv = ntuple(i -> LinRange(0, Li[i], n), Val(3))
    # grid spacing
    dxi = dx, dy, dz = ntuple(i -> xvi[i][2] - xvi[i][1], Val(3))
    # nodal centers
    xci = xc, yc, zc = ntuple(i -> LinRange(0 + dxi[i] / 2, Li[i] - dxi[i] / 2, ni[i]), Val(3))
    # staggered grid velocity nodal locations
    grid_vx = xv, expand_range(yc), expand_range(zc)
    grid_vy = expand_range(xc), yv, expand_range(zc)
    grid_vz = expand_range(xc), expand_range(yc), zv
    grid_vel = grid_vx, grid_vy, grid_vz

    particles = JP3.init_particles(backend, nxcell, max_xcell, min_xcell, grid_vel...)
    phases, pT = JP3.init_cell_arrays(particles, Val(2))
    phase_ratios = JP3.PhaseRatios(backend, 2, ni)
    particle_args = (phases, pT)
    particle_args_reduced = (phases,)
    particle_args_kwarg = (phases,)
    initial_elevation = Ly / 2
    chain = JP2.init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, initial_elevation)
    it = 500
    @views particles.index.data[:, 1:3, 1] .= 1.0
    @views particles.index.data[:, 4:6, 1] .= 0.0

    JP3.checkpointing_particles(pwd(), particles; phases = phases, phase_ratios = phase_ratios, particle_args = particle_args, particle_args_reduced = particle_args_reduced, particle_args_kwarg = particle_args_kwarg, it = it)

    # test type conversion
    @test eltype(eltype(Array(phases))) === Float64
    @test eltype(eltype(Array(Float64, phases))) === Float64
    @test eltype(eltype(Array(Float32, phases))) === Float32
    @test eltype(eltype(Array(particles).coords[1].data)) === Float64
    @test eltype(eltype(Array(Float64, particles).coords[1].data)) === Float64
    @test eltype(eltype(Array(Float32, particles).coords[1].data)) === Float32
    @test eltype(eltype(Array(phase_ratios).vertex.data)) === Float64
    @test eltype(eltype(Array(Float64, phase_ratios).vertex.data)) === Float64
    @test eltype(eltype(Array(Float32, phase_ratios).vertex.data)) === Float32
    @test eltype(eltype(Array(particles).index.data)) === Bool
    @test eltype(eltype(Array(Float32, particles).index.data)) === Bool
    @test eltype(eltype(Array(Float64, particles).index.data)) === Bool

    jldsave(
        "particles.jld2";
        particles = Array(particles),
        phases = Array(phases),
        phase_ratios = Array(phase_ratios)
    )

    data = load("particles.jld2")
    particles2 = data["particles"]
    phases2 = data["phases"]
    phase_ratios2 = data["phase_ratios"]

    data1 = load("particles_checkpoint.jld2")
    particles3 = data1["particles"]
    phases3 = data1["phases"]
    phase_ratios3 = data1["phase_ratios"]
    chain3 = data1["chain"]
    particle_args3 = data1["particle_args"]
    particle_args_reduced3 = data1["particle_args_reduced"]
    particle_args_kwarg3 = data1["particle_args_kwarg"]
    it1 = data1["it"]

    @test chain3 isa Nothing
    @test particle_args3 isa Tuple
    @test particle_args_reduced3 isa Tuple
    @test particle_args_kwarg3 isa Tuple
    @test Array(particles).coords[1].data == particles2.coords[1].data
    @test Array(particles).coords[2].data == particles2.coords[2].data
    @test Array(particles).index.data == particles2.index.data
    @test Array(phase_ratios).center.data == phase_ratios2.center.data
    @test Array(phase_ratios).vertex.data == phase_ratios2.vertex.data
    @test Array(phases).data == phases2.data
    @test size(Array(particles).coords[1].data) == size(particles2.coords[1].data)
    @test size(Array(particles).coords[2].data) == size(particles2.coords[2].data)
    @test size(Array(particles).index.data) == size(particles2.index.data)
    @test size(Array(phase_ratios).center.data) == size(phase_ratios2.center.data)
    @test size(Array(phase_ratios).vertex.data) == size(phase_ratios2.vertex.data)
    @test size(Array(phases).data) == size(phases2.data)

    @test Array(particles).coords[1].data == particles3.coords[1].data
    @test Array(particles).coords[2].data == particles3.coords[2].data
    @test Array(particles).index.data == particles3.index.data
    @test Array(phase_ratios).center.data == phase_ratios3.center.data
    @test Array(phase_ratios).vertex.data == phase_ratios3.vertex.data
    @test Array(phases).data == phases3.data
    @test size(Array(particles).coords[1].data) == size(particles3.coords[1].data)
    @test size(Array(particles).coords[2].data) == size(particles3.coords[2].data)
    @test size(Array(particles).index.data) == size(particles3.index.data)
    @test size(Array(phase_ratios).center.data) == size(phase_ratios3.center.data)
    @test size(Array(phase_ratios).vertex.data) == size(phase_ratios3.vertex.data)
    @test size(Array(phases).data) == size(phases3.data)

    # Test on GPU card, if available
    isCUDA = isdefined(Main, :CUDA)
    isAMDGPU = isdefined(Main, :AMDGPU)

    if isCUDA || isAMDGPU
        T = isCUDA ? CuArray : ROCArray
        Backend = isCUDA ? CUDABackend : AMDGPUBackend

        particles2 = Array(particles)
        phases2 = Array(phases)
        phase_ratios2 = Array(phase_ratios)
        particles_gpu = T(particles2)
        phase_ratios_gpu = T(phase_ratios2)
        phases_gpu = T(phases2)
        particles_gpu2 = T(particles3)
        phase_ratios_gpu2 = T(phase_ratios3)
        phases_gpu2 = T(phases3)
        particle_args_gpu2 = T.(particle_args)
        particle_args_reduced_gpu2 = T.(particle_args_reduced)

        @test particles_gpu isa JustPIC.Particles{Backend}
        @test phase_ratios_gpu isa JustPIC.PhaseRatios{Backend}
        @test last(typeof(phases_gpu).parameters) <: T{Float64, 3}
        @test size(particles_gpu.coords[1].data) == size(permutedims(particles.coords[1].data, (3, 2, 1)))
        @test size(particles_gpu.coords[2].data) == size(permutedims(particles.coords[2].data, (3, 2, 1)))
        @test size(particles_gpu.index.data) == size(permutedims(particles.index.data, (3, 2, 1)))
        @test size(phase_ratios_gpu.center.data) == size(permutedims(phase_ratios.center.data, (3, 2, 1)))
        @test size(phase_ratios_gpu.vertex.data) == size(permutedims(phase_ratios.vertex.data, (3, 2, 1)))
        @test size(phases_gpu.data) == size(permutedims(phases.data, (3, 2, 1)))

        @test particles_gpu2 isa JustPIC.Particles{Backend}
        @test phase_ratios_gpu2 isa JustPIC.PhaseRatios{Backend}
        @test particle_args_gpu2 isa Tuple
        @test particle_args_reduced_gpu2 isa Tuple
        @test last(typeof(phases_gpu2).parameters) <: T{Float64, 3}
        @test size(particles_gpu2.coords[1].data) == size(permutedims(particles.coords[1].data, (3, 2, 1)))
        @test size(particles_gpu2.coords[2].data) == size(permutedims(particles.coords[2].data, (3, 2, 1)))
        @test size(particles_gpu2.index.data) == size(permutedims(particles.index.data, (3, 2, 1)))
        @test size(phase_ratios_gpu2.center.data) == size(permutedims(phase_ratios.center.data, (3, 2, 1)))
        @test size(phase_ratios_gpu2.vertex.data) == size(permutedims(phase_ratios.vertex.data, (3, 2, 1)))
        @test size(phases_gpu2.data) == size(permutedims(phases.data, (3, 2, 1)))

        # test type conversion
        @test eltype(eltype(T(phases))) === Float64
        @test eltype(eltype(T(Float64, phases))) === Float64
        @test eltype(eltype(T(Float32, phases))) === Float32
        @test eltype(eltype(T(particles).coords[1].data)) === Float64
        @test eltype(eltype(T(Float64, particles).coords[1].data)) === Float64
        @test eltype(eltype(T(Float32, particles).coords[1].data)) === Float32
        @test eltype(eltype(T(phase_ratios).vertex.data)) === Float64
        @test eltype(eltype(T(Float64, phase_ratios).vertex.data)) === Float64
        @test eltype(eltype(T(Float32, phase_ratios).vertex.data)) === Float32
        @test eltype(eltype(T(particles).index.data)) === Bool
        @test eltype(eltype(T(Float32, particles).index.data)) === Bool
        @test eltype(eltype(T(Float64, particles).index.data)) === Bool
    end

    rm("particles.jld2") # cleanup
    rm("particles_checkpoint.jld2") # cleanup
end
