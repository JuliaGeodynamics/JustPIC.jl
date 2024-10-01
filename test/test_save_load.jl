using JLD2, JustPIC
import JustPIC._2D as JP2
import JustPIC._3D as JP3
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
    @test eltype(eltype(JP2.Array(phases)))                            === Float64
    @test eltype(eltype(JP2.Array(Float64, phases)))                   === Float64
    @test eltype(eltype(JP2.Array(Float32, phases)))                   === Float32
    @test eltype(eltype(JP2.Array(particles).coords[1].data))          === Float64
    @test eltype(eltype(JP2.Array(Float64, particles).coords[1].data)) === Float64
    @test eltype(eltype(JP2.Array(Float32, particles).coords[1].data)) === Float32
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

    if isdefined(Main, :CUDA)
        particles_cuda    = CuArray(particles2)
        phase_ratios_cuda = CuArray(phase_ratios2)
        phases_cuda       = CuArray(phases2);

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

    elseif isdefined(Main, :AMDGPU)
        particles_amdgpu    = ROCArray(particles2)
        phase_ratios_amdgpu = ROCArray(phase_ratios2)
        phases_amdgpu       = ROCArray(phases2)

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

        # test type conversion
        @test eltype(eltype(ROCArrayArray(phases)))                            === Float64
        @test eltype(eltype(ROCArrayArray(Float64, phases)))                   === Float64
        @test eltype(eltype(ROCArrayArray(Float32, phases)))                   === Float32
        @test eltype(eltype(ROCArrayArray(particles).coords[1].data))          === Float64
        @test eltype(eltype(ROCArrayArray(Float64, particles).coords[1].data)) === Float64
        @test eltype(eltype(ROCArrayArray(Float32, particles).coords[1].data)) === Float32
        @test eltype(eltype(ROCArrayArray(phase_ratios).vertex.data))          === Float64
        @test eltype(eltype(ROCArrayArray(Float64, phase_ratios).vertex.data)) === Float64
        @test eltype(eltype(ROCArrayArray(Float32, phase_ratios).vertex.data)) === Float32
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

    if isdefined(Main, :CUDA)
        particles_cuda    = CuArray(particles2)
        phase_ratios_cuda = CuArray(phase_ratios2)
        phases_cuda       = CuArray(phases2);

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

    elseif isdefined(Main, :AMDGPU)
        particles_amdgpu    = JP3.ROCArray(particles2)
        phase_ratios_amdgpu = JP3.ROCArray(phase_ratios2)
        phases_amdgpu       = JP3.ROCArray(phases2)

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

        # test type conversion
        @test eltype(eltype(JP3.ROCArrayArray(phases)))                            === Float64
        @test eltype(eltype(JP3.ROCArrayArray(Float64, phases)))                   === Float64
        @test eltype(eltype(JP3.ROCArrayArray(Float32, phases)))                   === Float32
        @test eltype(eltype(JP3.ROCArrayArray(particles).coords[1].data))          === Float64
        @test eltype(eltype(JP3.ROCArrayArray(Float64, particles).coords[1].data)) === Float64
        @test eltype(eltype(JP3.ROCArrayArray(Float32, particles).coords[1].data)) === Float32
        @test eltype(eltype(JP3.ROCArrayArray(phase_ratios).vertex.data))          === Float64
        @test eltype(eltype(JP3.ROCArrayArray(Float64, phase_ratios).vertex.data)) === Float64
        @test eltype(eltype(JP3.ROCArrayArray(Float32, phase_ratios).vertex.data)) === Float32
    end
    
    rm("particles.jld2") # cleanup
end
