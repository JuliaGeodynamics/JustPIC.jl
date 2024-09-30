using CUDA
using JLD2, JustPIC, JustPIC._2D
backend = CUDABackend
# backend = JustPIC.CPUBackend
using Test
@testset "Save and load" begin
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 6, 6, 6
    n  = 128
    nx = ny = n-1
    ni = nx, ny
    Lx = Ly = 1.0
    # nodal vertices
    xvi = xv, yv = range(0, Lx, length=n), range(0, Ly, length=n)

    particles    = init_particles(backend, nxcell, max_xcell, min_xcell, xvi...,);
    phases,      = init_cell_arrays(particles, Val(1));
    phase_ratios = PhaseRatios(backend, 2, ni);

    # test type conversion
    @test eltype(eltype(Array(phases)))                            === Float64
    @test eltype(eltype(Array(Float64, phases)))                   === Float64
    @test eltype(eltype(Array(Float32, phases)))                   === Float32
    @test eltype(eltype(Array(particles).coords[1].data))          === Float64
    @test eltype(eltype(Array(Float64, particles).coords[1].data)) === Float64
    @test eltype(eltype(Array(Float32, particles).coords[1].data)) === Float32
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

    @test Array(particles).coords[1].data        == particles2.coords[1].data
    @test Array(particles).coords[2].data        == particles2.coords[2].data
    @test Array(particles).index.data            == particles2.index.data
    @test Array(phase_ratios).center.data        == phase_ratios2.center.data
    @test Array(phase_ratios).vertex.data        == phase_ratios2.vertex.data
    @test Array(phases).data                     == phases2.data
    @test size(Array(particles).coords[1].data)  == size(particles2.coords[1].data)
    @test size(Array(particles).coords[2].data)  == size(particles2.coords[2].data)
    @test size(Array(particles).index.data)      == size(particles2.index.data)
    @test size(Array(phase_ratios).center.data)  == size(phase_ratios2.center.data)
    @test size(Array(phase_ratios).vertex.data)  == size(phase_ratios2.vertex.data)
    @test size(Array(phases).data)               == size(phases2.data)

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
