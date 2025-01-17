@static if ENV["JULIA_JUSTPIC_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTPIC_BACKEND"] === "CUDA"
    using CUDA
end

using JLD2, JustPIC, Test
import JustPIC._2D as JP2
import JustPIC._3D as JP3

const backend = JustPIC.CPUBackend

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
    phase_ratios = JP2.PhaseRatios(backend, 2, ni);
    @views particles.index.data[:, 1:3, 1] .= 1.0
    @views particles.index.data[:, 4:6, 1] .= 0.0

    # test type conversion
    @test eltype(eltype(Array(phases)))                            === Float64
    @test eltype(eltype(Array(Float64, phases)))                   === Float64
    @test eltype(eltype(Array(Float32, phases)))                   === Float32
    @test eltype(eltype(Array(particles).coords[1].data))          === Float64
    @test eltype(eltype(Array(Float64, particles).coords[1].data)) === Float64
    @test eltype(eltype(Array(Float32, particles).coords[1].data)) === Float32
    @test eltype(eltype(Array(particles).index.data))              === Bool
    @test eltype(eltype(Array(Float32, particles).index.data))     === Bool
    @test eltype(eltype(Array(Float64, particles).index.data))     === Bool
    @test eltype(eltype(Array(phase_ratios).vertex.data))          === Float64
    @test eltype(eltype(Array(Float64, phase_ratios).vertex.data)) === Float64
    @test eltype(eltype(Array(Float32, phase_ratios).vertex.data)) === Float32

    jldsave(
        "particles.jld2"; 
        particles    = Array(particles), 
        phases       = Array(phases), 
        phase_ratios = Array(phase_ratios)
    )

    data          = load("particles.jld2")
    particles2    = data["particles"]
    phases2       = data["phases"]
    phase_ratios2 = data["phase_ratios"]

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

    
    # Test on GPU card, if available
    isCUDA = isdefined(Main, :CUDA)
    isAMDGPU = isdefined(Main, :AMDGPU)

    if isCUDA || isAMDGPU
        T       = isCUDA ? :CuArray : :ROCArray
        Backend = isCUDA ? :CUDABackend : :AMDGPUBackend

        @eval begin
            particles_gpu    = $T(particles2)
            phase_ratios_gpu = $T(phase_ratios2)
            phases_gpu       = $T(phases2);

            @test particles_gpu                       isa JustPIC.Particles{$Backend}
            @test phase_ratios_gpu                    isa JustPIC.PhaseRatios{$Backend}
            @test last(typeof(phases_gpu).parameters) <: $T{Float64, 3}
            @test size(particles_gpu.coords[1].data)  == size(permutedims(particles.coords[1].data, (3, 2, 1)))
            @test size(particles_gpu.coords[2].data)  == size(permutedims(particles.coords[2].data, (3, 2, 1)))
            @test size(particles_gpu.index.data)      == size(permutedims(particles.index.data, (3, 2, 1)))
            @test size(phase_ratios_gpu.center.data)  == size(permutedims(phase_ratios.center.data, (3, 2, 1)))
            @test size(phase_ratios_gpu.vertex.data)  == size(permutedims(phase_ratios.vertex.data, (3, 2, 1)))
            @test size(phases_gpu.data)               == size(permutedims(phases.data, (3, 2, 1)))

            # test type conversion
            @test eltype(eltype($T(phases)))                            === Float64
            @test eltype(eltype($T(Float64, phases)))                   === Float64
            @test eltype(eltype($T(Float32, phases)))                   === Float32
            @test eltype(eltype($T(particles).coords[1].data))          === Float64
            @test eltype(eltype($T(Float64, particles).coords[1].data)) === Float64
            @test eltype(eltype($T(Float32, particles).coords[1].data)) === Float32
            @test eltype(eltype($T(phase_ratios).vertex.data))          === Float64
            @test eltype(eltype($T(Float64, phase_ratios).vertex.data)) === Float64
            @test eltype(eltype($T(Float32, phase_ratios).vertex.data)) === Float32
            @test eltype(eltype($T(particles).index.data))              === Bool
            @test eltype(eltype($T(Float32, particles).index.data))     === Bool
            @test eltype(eltype($T(Float64, particles).index.data))     === Bool
        end
    end
    
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

     # Test on GPU card, if available
     isCUDA = isdefined(Main, :CUDA)
     isAMDGPU = isdefined(Main, :AMDGPU)
 
     if isCUDA || isAMDGPU
        T       = isCUDA ? :CuArray : :ROCArray
        Backend = isCUDA ? :CUDABackend : :AMDGPUBackend

        @eval begin
            particles_gpu    = $T(particles2)
            phase_ratios_gpu = $T(phase_ratios2)
            phases_gpu       = $T(phases2);

            @test particles_gpu                       isa JustPIC.Particles{$Backend}
            @test phase_ratios_gpu                    isa JustPIC.PhaseRatios{$Backend}
            @test last(typeof(phases_gpu).parameters) <: $T{Float64, 3}
            @test size(particles_gpu.coords[1].data)  == size(permutedims(particles.coords[1].data, (3, 2, 1)))
            @test size(particles_gpu.coords[2].data)  == size(permutedims(particles.coords[2].data, (3, 2, 1)))
            @test size(particles_gpu.index.data)      == size(permutedims(particles.index.data, (3, 2, 1)))
            @test size(phase_ratios_gpu.center.data)  == size(permutedims(phase_ratios.center.data, (3, 2, 1)))
            @test size(phase_ratios_gpu.vertex.data)  == size(permutedims(phase_ratios.vertex.data, (3, 2, 1)))
            @test size(phases_gpu.data)               == size(permutedims(phases.data, (3, 2, 1)))

            # test type conversion
            @test eltype(eltype($T(phases)))                            === Float64
            @test eltype(eltype($T(Float64, phases)))                   === Float64
            @test eltype(eltype($T(Float32, phases)))                   === Float32
            @test eltype(eltype($T(particles).coords[1].data))          === Float64
            @test eltype(eltype($T(Float64, particles).coords[1].data)) === Float64
            @test eltype(eltype($T(Float32, particles).coords[1].data)) === Float32
            @test eltype(eltype($T(phase_ratios).vertex.data))          === Float64
            @test eltype(eltype($T(Float64, phase_ratios).vertex.data)) === Float64
            @test eltype(eltype($T(Float32, phase_ratios).vertex.data)) === Float32
            @test eltype(eltype($T(particles).index.data))              === Bool
            @test eltype(eltype($T(Float32, particles).index.data))     === Bool
            @test eltype(eltype($T(Float64, particles).index.data))     === Bool
        end
     end
    
    rm("particles.jld2") # cleanup
end
