@static if ENV["JULIA_JUSTPIC_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTPIC_BACKEND"] === "CUDA"
    using CUDA
end

using JustPIC
import JustPIC._2D as JP2
import JustPIC._3D as JP3

using CellArrays, Test, JLD2

const backend = @static if ENV["JULIA_JUSTPIC_BACKEND"] === "AMDGPU"
    JustPIC.AMDGPUBackend
elseif ENV["JULIA_JUSTPIC_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPUBackend
end
@testset "Save and load 2D" begin
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 6, 6, 6
    n  = 64
    nx = ny = n-1
    ni = nx, ny
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = range(0, Lx, length=n), range(0, Ly, length=n)

    particles    = JP2.init_particles(backend, nxcell, max_xcell, min_xcell, xvi...,);
    phases,      = JP2.init_cell_arrays(particles, Val(1));
    particle_args = (phases,)
    phase_ratios = JP2.PhaseRatios(backend, 2, ni);
    initial_elevation = Ly/2
    chain             = JP2.init_markerchain(backend, nxcell, min_xcell, max_xcell, xv, initial_elevation);
    @views particles.index.data[:, 1:3, 1] .= 1.0
    @views particles.index.data[:, 4:6, 1] .= 0.0

    # test type conversion
    @test eltype(eltype(JP2.Array(phases)))                            === Float64
    @test eltype(eltype(JP2.Array(Float64, phases)))                   === Float64
    @test eltype(eltype(JP2.Array(Float32, phases)))                   === Float32
    @test eltype(eltype(JP2.Array(particles).coords[1].data))          === Float64
    @test eltype(eltype(JP2.Array(Float64, particles).coords[1].data)) === Float64
    @test eltype(eltype(JP2.Array(Float32, particles).coords[1].data)) === Float32
    @test eltype(eltype(JP2.Array(particles).index.data))              === Bool
    @test eltype(eltype(JP2.Array(Float32, particles).index.data))     === Bool
    @test eltype(eltype(JP2.Array(Float64, particles).index.data))     === Bool
    @test eltype(eltype(JP2.Array(phase_ratios).vertex.data))          === Float64
    @test eltype(eltype(JP2.Array(Float64, phase_ratios).vertex.data)) === Float64
    @test eltype(eltype(JP2.Array(Float32, phase_ratios).vertex.data)) === Float32

    jldsave(
        "particles.jld2";
        particles    = JP2.Array(particles),
        phases       = JP2.Array(phases),
        phase_ratios = JP2.Array(phase_ratios)
    )

    data          = load("particles.jld2")
    particles2    = data["particles"]
    phases2       = data["phases"]
    phase_ratios2 = data["phase_ratios"]

    JP2.checkpointing_particles(@__DIR__, particles, phases, phase_ratios; chain=chain, particle_args=particle_args)

    data1          = load("particles_checkpoint.jld2")
    particles3    = data1["particles"]
    phases3       = data1["phases"]
    phase_ratios3 = data1["phase_ratios"]
    chain3         = data1["chain"]
    particle_args3 = data1["particle_args"]

    @test JP2.Array(particles).coords[1].data        == particles2.coords[1].data
    @test JP2.Array(particles).coords[2].data        == particles2.coords[2].data
    @test JP2.Array(particles).index.data            == particles2.index.data
    @test JP2.Array(phase_ratios).center.data        == phase_ratios2.center.data
    @test JP2.Array(phase_ratios).vertex.data        == phase_ratios2.vertex.data
    @test JP2.Array(phases).data                     == phases2.data
    @test size(JP2.Array(particles).coords[1].data)  == size(particles2.coords[1].data)
    @test size(JP2.Array(particles).coords[2].data)  == size(particles2.coords[2].data)
    @test size(JP2.Array(particles).index.data)      == size(particles2.index.data)
    @test size(JP2.Array(phase_ratios).center.data)  == size(phase_ratios2.center.data)
    @test size(JP2.Array(phase_ratios).vertex.data)  == size(phase_ratios2.vertex.data)
    @test size(JP2.Array(phases).data)               == size(phases2.data)

    @test chain3                                      isa JustPIC.MarkerChain{JustPIC.CPUBackend}
    @test particle_args3                              isa Tuple
    @test JP2.Array(particles).coords[1].data        == particles3.coords[1].data
    @test JP2.Array(particles).coords[2].data        == particles3.coords[2].data
    @test JP2.Array(particles).index.data            == particles3.index.data
    @test JP2.Array(phase_ratios).center.data        == phase_ratios3.center.data
    @test JP2.Array(phase_ratios).vertex.data        == phase_ratios3.vertex.data
    @test JP2.Array(phases).data                     == phases3.data
    @test size(JP2.Array(particles).coords[1].data)  == size(particles3.coords[1].data)
    @test size(JP2.Array(particles).coords[2].data)  == size(particles3.coords[2].data)
    @test size(JP2.Array(particles).index.data)      == size(particles3.index.data)
    @test size(JP2.Array(phase_ratios).center.data)  == size(phase_ratios3.center.data)
    @test size(JP2.Array(phase_ratios).vertex.data)  == size(phase_ratios3.vertex.data)
    @test size(JP2.Array(phases).data)               == size(phases3.data)

    if isdefined(Main, :CUDA)
        particles_cuda    = CuArray(particles2)
        phase_ratios_cuda = CuArray(phase_ratios2)
        phases_cuda       = CuArray(phases2);

        particles_cuda2    = CuArray(particles3)
        phase_ratios_cuda2 = CuArray(phase_ratios3)
        phases_cuda2       = CuArray(phases3);
        chain_cuda         = CuArray(chain)
        particle_args_cuda = CuArray.(particle_args)

        @test particles_cuda                       isa JustPIC.Particles{CUDABackend}
        @test phase_ratios_cuda                    isa JustPIC.PhaseRatios{CUDABackend}
        @test chain_cuda                           isa JustPIC.MarkerChain{CUDABackend}
        @test particle_args_cuda                   isa Tuple
        @test last(typeof(phases_cuda).parameters) <: CuArray{Float64, 3}
        @test typeof(phases_cuda)                  == typeof(phases)
        @test size(particles_cuda.coords[1].data)  == size(particles.coords[1].data)
        @test size(particles_cuda.coords[2].data)  == size(particles.coords[2].data)
        @test size(particles_cuda.index.data)      == size(particles.index.data)
        @test size(phase_ratios_cuda.center.data)  == size(phase_ratios.center.data)
        @test size(phase_ratios_cuda.vertex.data)  == size(phase_ratios.vertex.data)
        @test size(phases_cuda.data)               == size(phases.data)

        @test particles_cuda2                       isa JustPIC.Particles{CUDABackend}
        @test phase_ratios_cuda2                    isa JustPIC.PhaseRatios{CUDABackend}
        @test last(typeof(phases_cuda2).parameters) <: CuArray{Float64, 3}
        @test typeof(phases_cuda2)                  == typeof(phases)
        @test size(particles_cuda2.coords[1].data)  == size(particles.coords[1].data)
        @test size(particles_cuda2.coords[2].data)  == size(particles.coords[2].data)
        @test size(particles_cuda2.index.data)      == size(particles.index.data)
        @test size(phase_ratios_cuda2.center.data)  == size(phase_ratios.center.data)
        @test size(phase_ratios_cuda2.vertex.data)  == size(phase_ratios.vertex.data)
        @test size(phases_cuda2.data)               == size(phases.data)

        # test type conversion
        @test eltype(eltype(CuArray(phases)))                            === Float64
        @test eltype(eltype(CuArray(Float64, phases)))                   === Float64
        @test eltype(eltype(CuArray(Float32, phases)))                   === Float32
        @test eltype(eltype(CuArray(particles).coords[1].data))          === Float64
        @test eltype(eltype(CuArray(Float64, particles).coords[1].data)) === Float64
        @test eltype(eltype(CuArray(Float32, particles).coords[1].data)) === Float32
        @test eltype(eltype(CuArray(phase_ratios).vertex.data))          === Float64
        @test eltype(eltype(CuArray(Float64, phase_ratios).vertex.data)) === Float64
        @test eltype(eltype(CuArray(Float32, phase_ratios).vertex.data)) === Float32
        @test eltype(eltype(CuArray(particles).index.data))              === Bool
        @test eltype(eltype(CuArray(Float32, particles).index.data))     === Bool
        @test eltype(eltype(CuArray(Float64, particles).index.data))     === Bool

    elseif isdefined(Main, :AMDGPU)
        particles_amdgpu    = ROCArray(particles2)
        phase_ratios_amdgpu = ROCArray(phase_ratios2)
        phases_amdgpu       = ROCArray(phases2)

        particles_amdgpu2    = ROCArray(particles2)
        phase_ratios_amdgpu2 = ROCArray(phase_ratios2)
        phases_amdgpu2       = ROCArray(phases2)
        chain_amdgpu         = ROCArray(chain)
        partile_args_ampgpu  = ROCArray.(particle_args)

        @test particles_amdgpu2                       isa JustPIC.Particles{AMDGPUBackend}
        @test phase_ratios_amdgpu2                    isa JustPIC.PhaseRatios{AMDGPUBackend}
        @test chain_amdgpu                            isa JustPIC.MarkerChain{AMDGPUBackend}
        @test partiles_args_ampgpu                    isa Tuple
        @test last(typeof(phases_amdgpu2).parameters) <: ROCArray{Float64, 3}
        @test typeof(phases_amdgpu2)                  == typeof(phases)
        @test size(particles_amdgpu2.coords[1].data)  == size(particles.coords[1].data)
        @test size(particles_amdgpu2.coords[2].data)  == size(particles.coords[2].data)
        @test size(particles_amdgpu2.index.data)      == size(particles.index.data)
        @test size(phase_ratios_amdgpu2.center.data)  == size(phase_ratios.center.data)
        @test size(phase_ratios_amdgpu2.vertex.data)  == size(phase_ratios.vertex.data)
        @test size(phases_amdgpu2.data)               == size(phases.data)

        # test type conversion
        @test eltype(eltype(ROCArray(phases)))                            === Float64
        @test eltype(eltype(ROCArray(Float64, phases)))                   === Float64
        @test eltype(eltype(ROCArray(Float32, phases)))                   === Float32
        @test eltype(eltype(ROCArray(particles).coords[1].data))          === Float64
        @test eltype(eltype(ROCArray(Float64, particles).coords[1].data)) === Float64
        @test eltype(eltype(ROCArray(Float32, particles).coords[1].data)) === Float32
        @test eltype(eltype(ROCArray(phase_ratios).vertex.data))          === Float64
        @test eltype(eltype(ROCArray(Float64, phase_ratios).vertex.data)) === Float64
        @test eltype(eltype(ROCArray(Float32, phase_ratios).vertex.data)) === Float32
        @test eltype(eltype(ROCArray(particles).index.data))              === Bool
        @test eltype(eltype(ROCArray(Float32, particles).index.data))     === Bool
        @test eltype(eltype(ROCArray(Float64, particles).index.data))     === Bool
    end

    rm("particles_checkpoint.jld2") # cleanup
    rm("particles.jld2") # cleanup
end

@testset "Save and load 3D" begin
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 6, 6, 6
    n  = 64
    nx = ny = nz = n-1
    ni = nx, ny, nz
    Lx = Ly = Lz = 1.0
    # nodal vertices
    xvi = xv, yv, zv = range(0, Lx, length=n), range(0, Ly, length=n), range(0, Lz, length=n)

    particles    = JP3.init_particles(backend, nxcell, max_xcell, min_xcell, xvi...,);
    phases,      = JP3.init_cell_arrays(particles, Val(1));
    phase_ratios = JP3.PhaseRatios(backend, 2, ni);

    # test type conversion
    @test eltype(eltype(JP3.Array(phases)))                            === Float64
    @test eltype(eltype(JP3.Array(Float64, phases)))                   === Float64
    @test eltype(eltype(JP3.Array(Float32, phases)))                   === Float32
    @test eltype(eltype(JP3.Array(particles).coords[1].data))          === Float64
    @test eltype(eltype(JP3.Array(Float64, particles).coords[1].data)) === Float64
    @test eltype(eltype(JP3.Array(Float32, particles).coords[1].data)) === Float32
    @test eltype(eltype(JP3.Array(phase_ratios).vertex.data))          === Float64
    @test eltype(eltype(JP3.Array(Float64, phase_ratios).vertex.data)) === Float64
    @test eltype(eltype(JP3.Array(Float32, phase_ratios).vertex.data)) === Float32
    @test eltype(eltype(JP3.Array(particles).index.data))              === Bool
    @test eltype(eltype(JP3.Array(Float32, particles).index.data))     === Bool
    @test eltype(eltype(JP3.Array(Float64, particles).index.data))     === Bool

    particles.index.data[:, 1:3, 1] .= 1.0
    particles.index.data[:, 4:6, 1] .= 0.0

    jldsave(
        "particles.jld2";
        particles    = JP3.Array(particles),
        phases       = JP3.Array(phases),
        phase_ratios = JP3.Array(phase_ratios)
    )

    data          = load("particles.jld2")
    particles2    = data["particles"]
    phases2       = data["phases"]
    phase_ratios2 = data["phase_ratios"]

    JP3.checkpointing_particles(@__DIR__, particles, phases, phase_ratios)

    data1          = load("particles_checkpoint.jld2")
    particles3    = data1["particles"]
    phases3       = data1["phases"]
    phase_ratios3 = data1["phase_ratios"]

    @test JP3.Array(particles).coords[1].data        == particles2.coords[1].data
    @test JP3.Array(particles).coords[2].data        == particles2.coords[2].data
    @test JP3.Array(particles).index.data            == particles2.index.data
    @test JP3.Array(phase_ratios).center.data        == phase_ratios2.center.data
    @test JP3.Array(phase_ratios).vertex.data        == phase_ratios2.vertex.data
    @test JP3.Array(phases).data                     == phases2.data
    @test size(JP3.Array(particles).coords[1].data)  == size(particles2.coords[1].data)
    @test size(JP3.Array(particles).coords[2].data)  == size(particles2.coords[2].data)
    @test size(JP3.Array(particles).index.data)      == size(particles2.index.data)
    @test size(JP3.Array(phase_ratios).center.data)  == size(phase_ratios2.center.data)
    @test size(JP3.Array(phase_ratios).vertex.data)  == size(phase_ratios2.vertex.data)
    @test size(JP3.Array(phases).data)               == size(phases2.data)

    @test JP3.Array(particles).coords[1].data        == particles3.coords[1].data
    @test JP3.Array(particles).coords[2].data        == particles3.coords[2].data
    @test JP3.Array(particles).index.data            == particles3.index.data
    @test JP3.Array(phase_ratios).center.data        == phase_ratios3.center.data
    @test JP3.Array(phase_ratios).vertex.data        == phase_ratios3.vertex.data
    @test JP3.Array(phases).data                     == phases3.data
    @test size(JP3.Array(particles).coords[1].data)  == size(particles3.coords[1].data)
    @test size(JP3.Array(particles).coords[2].data)  == size(particles3.coords[2].data)
    @test size(JP3.Array(particles).index.data)      == size(particles3.index.data)
    @test size(JP3.Array(phase_ratios).center.data)  == size(phase_ratios3.center.data)
    @test size(JP3.Array(phase_ratios).vertex.data)  == size(phase_ratios3.vertex.data)
    @test size(JP3.Array(phases).data)               == size(phases3.data)

    if isdefined(Main, :CUDA)
        particles_cuda    = CuArray(particles2)
        phase_ratios_cuda = CuArray(phase_ratios2)
        phases_cuda       = CuArray(phases2);

        particles_cuda2    = CuArray(particles3)
        phase_ratios_cuda2 = CuArray(phase_ratios3)
        phases_cuda2       = CuArray(phases3);

        @test particles_cuda                       isa JustPIC.Particles{CUDABackend}
        @test phase_ratios_cuda                    isa JustPIC.PhaseRatios{CUDABackend}
        @test last(typeof(phases_cuda).parameters) <: CuArray{Float64, 3}
        @test typeof(phases_cuda)                  == typeof(phases)
        @test size(particles_cuda.coords[1].data)  == size(particles.coords[1].data)
        @test size(particles_cuda.coords[2].data)  == size(particles.coords[2].data)
        @test size(particles_cuda.index.data)      == size(particles.index.data)
        @test size(phase_ratios_cuda.center.data)  == size(phase_ratios.center.data)
        @test size(phase_ratios_cuda.vertex.data)  == size(phase_ratios.vertex.data)
        @test size(phases_cuda.data)               == size(phases.data)

        @test particles_cuda2                       isa JustPIC.Particles{CUDABackend}
        @test phase_ratios_cuda2                    isa JustPIC.PhaseRatios{CUDABackend}
        @test last(typeof(phases_cuda2).parameters) <: CuArray{Float64, 3}
        @test typeof(phases_cuda2)                  == typeof(phases)
        @test size(particles_cuda2.coords[1].data)  == size(particles.coords[1].data)
        @test size(particles_cuda2.coords[2].data)  == size(particles.coords[2].data)
        @test size(particles_cuda2.index.data)      == size(particles.index.data)
        @test size(phase_ratios_cuda2.center.data)  == size(phase_ratios.center.data)
        @test size(phase_ratios_cuda2.vertex.data)  == size(phase_ratios.vertex.data)
        @test size(phases_cuda2.data)               == size(phases.data)

        # test type conversion
        @test eltype(eltype(CuArray(phases)))                            === Float64
        @test eltype(eltype(CuArray(Float64, phases)))                   === Float64
        @test eltype(eltype(CuArray(Float32, phases)))                   === Float32
        @test eltype(eltype(CuArray(particles).coords[1].data))          === Float64
        @test eltype(eltype(CuArray(Float64, particles).coords[1].data)) === Float64
        @test eltype(eltype(CuArray(Float32, particles).coords[1].data)) === Float32
        @test eltype(eltype(CuArray(phase_ratios).vertex.data))          === Float64
        @test eltype(eltype(CuArray(Float64, phase_ratios).vertex.data)) === Float64
        @test eltype(eltype(CuArray(Float32, phase_ratios).vertex.data)) === Float32
        @test eltype(eltype(CuArray(particles).index.data))              === Bool
        @test eltype(eltype(CuArray(Float32, particles).index.data))     === Bool
        @test eltype(eltype(CuArray(Float64, particles).index.data))     === Bool

    elseif isdefined(Main, :AMDGPU)
        particles_amdgpu    = JP3.ROCArray(particles2)
        phase_ratios_amdgpu = JP3.ROCArray(phase_ratios2)
        phases_amdgpu       = JP3.ROCArray(phases2)

        particles_amdgpu2    = JP3.ROCArray(particles3)
        phase_ratios_amdgpu2 = JP3.ROCArray(phase_ratios3)
        phases_amdgpu2       = JP3.ROCArray(phases3)

        @test particles_amdgpu                       isa JustPIC.Particles{AMDGPUBackend}
        @test phase_ratios_amdgpu                    isa JustPIC.PhaseRatios{AMDGPUBackend}
        @test last(typeof(phases_amdgpu).parameters) <: ROCArray{Float64, 3}
        @test typeof(phases_amdgpu)                  == typeof(phases)
        @test size(particles_amdgpu.coords[1].data)  == size(particles.coords[1].data)
        @test size(particles_amdgpu.coords[2].data)  == size(particles.coords[2].data)
        @test size(particles_amdgpu.index.data)      == size(particles.index.data)
        @test size(phase_ratios_amdgpu.center.data)  == size(phase_ratios.center.data)
        @test size(phase_ratios_amdgpu.vertex.data)  == size(phase_ratios.vertex.data)
        @test size(phases_amdgpu.data)               == size(phases.data)

        @test particles_amdgpu2                       isa JustPIC.Particles{AMDGPUBackend}
        @test phase_ratios_amdgpu2                    isa JustPIC.PhaseRatios{AMDGPUBackend}
        @test last(typeof(phases_amdgpu2).parameters) <: ROCArray{Float64, 3}
        @test typeof(phases_amdgpu2)                  == typeof(phases)
        @test size(particles_amdgpu2.coords[1].data)  == size(particles.coords[1].data)
        @test size(particles_amdgpu2.coords[2].data)  == size(particles.coords[2].data)
        @test size(particles_amdgpu2.index.data)      == size(particles.index.data)
        @test size(phase_ratios_amdgpu2.center.data)  == size(phase_ratios.center.data)
        @test size(phase_ratios_amdgpu2.vertex.data)  == size(phase_ratios.vertex.data)
        @test size(phases_amdgpu2.data)               == size(phases.data)

        # test type conversion
        @test eltype(eltype(ROCArray(phases)))                            === Float64
        @test eltype(eltype(ROCArray(Float64, phases)))                   === Float64
        @test eltype(eltype(ROCArray(Float32, phases)))                   === Float32
        @test eltype(eltype(ROCArray(particles).coords[1].data))          === Float64
        @test eltype(eltype(ROCArray(Float64, particles).coords[1].data)) === Float64
        @test eltype(eltype(ROCArray(Float32, particles).coords[1].data)) === Float32
        @test eltype(eltype(ROCArray(phase_ratios).vertex.data))          === Float64
        @test eltype(eltype(ROCArray(Float64, phase_ratios).vertex.data)) === Float64
        @test eltype(eltype(ROCArray(Float32, phase_ratios).vertex.data)) === Float32
        @test eltype(eltype(ROCArray(particles).index.data))              === Bool
        @test eltype(eltype(ROCArray(Float32, particles).index.data))     === Bool
        @test eltype(eltype(ROCArray(Float64, particles).index.data))     === Bool

    end

    rm("particles.jld2") # cleanup
    rm("particles_checkpoint.jld2") # cleanup
end
