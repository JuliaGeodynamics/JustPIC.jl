using JLD2, JustPIC, JustPIC._2D

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

    @test Array(particles.coords[1].data) == particles2.coords[1].data
    @test Array(particles.coords[2].data) == particles2.coords[2].data
    @test Array(particles.index.data)     == particles2.index.data
    @test Array(phase_ratios.center.data) == phase_ratios2.center.data
    @test Array(phase_ratios.vertex.data) == phase_ratios2.vertex.data
    @test Array(phases.data)              == phases2.data

    if isdefined(Main, :CUDA)
        particles_cuda    = CuArray(particles2)
        phase_ratios_cuda = CuArray(phase_ratios2)
        phases_cuda       = CuArray(phases2)
        
        @test particles_cuda    isa JustPIC.Particles{CUDABackend} 
        @test phase_ratios_cuda isa JustPIC.PhaseRatios{CUDABackend} 
        @test phases_cuda       isa CuArray
        @test typeof(phases_cuda) == typeof(phases)

    elseif isdefined(Main, :AMDGPU)
        particles_amdgpu    = ROCArray(particles2)
        phase_ratios_amdgpu = ROCArray(phase_ratios2)
        phases_amdgpu       = ROCArray(phases2)

        @test particles_amdgpu    isa JustPIC.Particles{AMDGPUBackend} 
        @test phase_ratios_amdgpu isa JustPIC.PhaseRatios{AMDGPUBackend} 
        @test phases_amdgpu       isa ROCArray
        @test typeof(phases_amdgpu) == typeof(phases)

    end
end
