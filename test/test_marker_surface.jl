@static if ENV["JULIA_JUSTPIC_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTPIC_BACKEND"] === "CUDA"
    using CUDA
end

using JustPIC, JustPIC._3D, CellArrays, ParallelStencil, Test, LinearAlgebra

const backend = @static if ENV["JULIA_JUSTPIC_BACKEND"] === "AMDGPU"
    JustPIC.AMDGPUBackend
elseif ENV["JULIA_JUSTPIC_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPUBackend
end

function make_grid(; nx=8, ny=8, nz=8,
                     Lx=1.0, Ly=1.0, Lz=1.0)
    xv = LinRange(0.0, Lx, nx + 1)
    yv = LinRange(0.0, Ly, ny + 1)
    zv = LinRange(0.0, Lz, nz + 1)
    return xv, yv, zv
end

@testset "MarkerSurface — Initialization" begin
    xv, yv, zv = make_grid()

    @testset "Flat surface (scalar elevation)" begin
        surf = init_marker_surface(backend, xv, yv, 0.5)
        @test size(surf.topo) == (length(xv), length(yv))
        @test all(surf.topo .≈ 0.5)
        @test all(surf.topo0 .≈ 0.5)
        @test compute_avg_topo(surf) ≈ 0.5
        @test surf.air_phase == 0
    end

    @testset "Flat surface with options" begin
        surf = init_marker_surface(backend, xv, yv, 0.7; air_phase=2)
        @test surf.air_phase == 2
        @test compute_avg_topo(surf) ≈ 0.7
    end

    @testset "Variable initial topography" begin
        nx1, ny1 = length(xv), length(yv)
        z_init = [0.4 + 0.1 * sin(2π * xv[i]) * cos(2π * yv[j])
                  for i in 1:nx1, j in 1:ny1]
        surf = init_marker_surface(backend, xv, yv, z_init)
        @test Array(surf.topo) ≈ z_init
        @test Array(surf.topo0) ≈ z_init
        @test compute_avg_topo(surf) ≈ sum(z_init) / length(z_init)
    end

    @testset "set_topo_from_array!" begin
        surf = init_marker_surface(backend, xv, yv, 0.0)
        AT = TA(backend)
        z_new = AT(fill(0.3, length(xv), length(yv)))
        set_topo_from_array!(surf, z_new)
        @test all(Array(surf.topo) .≈ 0.3)
        @test all(Array(surf.topo0) .≈ 0.3)
        @test compute_avg_topo(surf) ≈ 0.3
    end
end

@testset "MarkerSurface — Triangle interpolation" begin
    # Test the helper directly

    @testset "Point inside triangle" begin
        # Equilateral-ish triangle in XY plane
        cx = (0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        cy = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        cz = (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        tri = (1, 2, 3)

        # Centroid of triangle
        xp = (0.0 + 1.0 + 0.5) / 3
        yp = (0.0 + 0.0 + 1.0) / 3
        ok, zp = _3D._interpolate_triangle(cx, cy, cz, tri, xp, yp)
        @test ok == true
        # At centroid, barycentric coords are (1/3, 1/3, 1/3)
        @test zp ≈ (1.0 + 2.0 + 3.0) / 3 atol = 1e-10
    end

    @testset "Point outside triangle" begin
        cx = (0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        cy = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        cz = (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        tri = (1, 2, 3)
        ok, _ = _3D._interpolate_triangle(cx, cy, cz, tri, -1.0, -1.0)
        @test ok == false
    end
end

@testset "MarkerSurface - Advection" begin
    @testset "MarkerSurface — Advection (zero velocity)" begin
        xv, yv, zv = make_grid()
        z0 = 0.5

        surf = init_marker_surface(backend, xv, yv, z0)
        dt = 0.01

        # Zero velocity field → topography should not change
        nx1, ny1 = length(xv), length(yv)
        fill!(surf.vx, 0.0)
        fill!(surf.vy, 0.0)
        fill!(surf.vz, 0.0)

        advect_surface_topo!(surf, dt)
        @test all(abs.(surf.topo .- z0) .< 1e-12)
    end

    @testset "MarkerSurface — Advection (uniform vertical velocity)" begin
        xv, yv, zv = make_grid()
        z0 = 0.5
        vz_val = 0.1
        dt = 0.1

        surf = init_marker_surface(backend, xv, yv, z0)

        # Set uniform vertical velocity, zero horizontal
        fill!(surf.vx, 0.0)
        fill!(surf.vy, 0.0)
        fill!(surf.vz, vz_val)

        advect_surface_topo!(surf, dt)

        # Expected: z0 + vz*dt = 0.5 + 0.01 = 0.51
        expected = z0 + vz_val * dt
        @test all(abs.(surf.topo .- expected) .< 1e-10)
    end

    @testset "MarkerSurface — Velocity interpolation" begin
        xv, yv, zv = make_grid(; nx=4, ny=4, nz=4)
        nx1, ny1 = length(xv), length(yv)

        surf = init_marker_surface(backend, xv, yv, 0.5)

        # Create constant velocity field Vz = 1.0 everywhere
        nxg, nyg, nzg = length(xv), length(yv), length(zv)
        AT = TA(backend)
        Vx = AT(zeros(nxg, nyg, nzg))
        Vy = AT(zeros(nxg, nyg, nzg))
        Vz = AT(ones(nxg, nyg, nzg))

        V = (Vx, Vy, Vz)
        xvi = (AT(collect(Float64, xv)), AT(collect(Float64, yv)), AT(collect(Float64, zv)))

        interpolate_velocity_to_surface_vertices!(surf, V, xvi)

        # With constant Vz=1, all surface vz should be 1
        @test all(abs.(Array(surf.vz) .- 1.0) .< 1e-10)
        @test all(abs.(Array(surf.vx)) .< 1e-10)
        @test all(abs.(Array(surf.vy)) .< 1e-10)
    end

    @testset "MarkerSurface — Full advection pipeline" begin
        xv, yv, zv = make_grid(; nx=8, ny=8, nz=8)
        z0 = 0.5

        surf = init_marker_surface(backend, xv, yv, z0)

        # Create a velocity field: uniform upward Vz=0.1
        nxg, nyg, nzg = length(xv), length(yv), length(zv)
        AT = TA(backend)
        Vx = AT(zeros(nxg, nyg, nzg))
        Vy = AT(zeros(nxg, nyg, nzg))
        Vz = AT(fill(0.1, nxg, nyg, nzg))
        V = (Vx, Vy, Vz)
        xvi = (AT(collect(Float64, xv)), AT(collect(Float64, yv)), AT(collect(Float64, zv)))

        dt = 0.1

        advect_marker_surface!(surf, V, xvi, dt)

        # Expected: z0 + 0.1 * 0.1 = 0.51
        @test all(abs.(Array(surf.topo) .- 0.51) .< 1e-8)
        @test abs(compute_avg_topo(surf) - 0.51) < 1e-8
    end

    @testset "MarkerSurface Extended — Advection convergence" begin
        # Test that uniform vertical uplift is exact regardless of resolution
        for nx in [4, 8, 16, 32]
            xv, yv, zv = make_grid(; nx=nx, ny=nx, nz=nx)
            z0 = 0.5
            vz_val = 0.2
            dt = 0.05
            surf = init_marker_surface(backend, xv, yv, z0)
            fill!(surf.vx, 0.0)
            fill!(surf.vy, 0.0)
            fill!(surf.vz, vz_val)
            advect_surface_topo!(surf, dt)
            expected = z0 + vz_val * dt
            err = maximum(abs.(surf.topo .- expected))
            @test err < 1e-12
        end
    end

    @testset "MarkerSurface Extended — Horizontal advection" begin
        # When there's only horizontal velocity (vx), the topography shape
        # should translate horizontally. Test with a sinusoidal surface.
        xv, yv, zv = make_grid(; nx=32, ny=4, nz=4)
        nx1, ny1 = length(xv), length(yv)

        z_init = [0.5 + 0.1 * sin(2π * xv[i]) for i in 1:nx1, j in 1:ny1]
        surf = init_marker_surface(backend, xv, yv, z_init)

        # Pure horizontal advection at small vx
        fill!(surf.vx, 0.01)
        fill!(surf.vy, 0.0)
        fill!(surf.vz, 0.0)

        dt = 0.001  # Small dt for accuracy
        advect_surface_topo!(surf, dt)

        # The surface should have shifted slightly; magnitude of change should be small
        max_change = maximum(abs.(Array(surf.topo) .- z_init))
        @test max_change > 0  # Something should change
        @test max_change < 0.01  # But not by much with small dt
    end

    @testset "MarkerSurface Extended — Background strain rate" begin
        xv, yv, zv = make_grid(; nx=8, ny=8, nz=8)
        z0 = 0.5

        surf = init_marker_surface(backend, xv, yv, z0)
        fill!(surf.vx, 0.0)
        fill!(surf.vy, 0.0)
        fill!(surf.vz, 0.0)

        # With zero velocity but non-zero background strain, the advection
        # should still produce a valid (flat) result since the target point
        # is found in the deformed grid
        dt = 0.1
        advect_surface_topo!(surf, dt; Exx=0.01, Eyy=0.01)
        advect_surface_topo!(surf, dt; Exx=0.01, Eyy=0.01)
        advect_surface_topo!(surf, dt; Exx=0.01, Eyy=0.01)
        # Flat topography with zero velocity → should remain flat
        @test all(abs.(Array(surf.topo) .- z0) .< 1e-10)
    end


    @testset "MarkerSurface Extended — Multiple timestep advection" begin
        xv, yv, zv = make_grid(; nx=8, ny=8, nz=16)
        z0 = 0.3
        vz_val = 0.1
        dt = 0.01
        nsteps = 10

        surf = init_marker_surface(backend, xv, yv, z0)

        nxg, nyg, nzg = length(xv), length(yv), length(zv)
        AT = TA(backend)
        Vx = AT(zeros(nxg, nyg, nzg))
        Vy = AT(zeros(nxg, nyg, nzg))
        Vz = AT(fill(vz_val, nxg, nyg, nzg))
        V = (Vx, Vy, Vz)
        xvi = (AT(collect(Float64, xv)), AT(collect(Float64, yv)), AT(collect(Float64, zv)))

        for _ in 1:nsteps
            advect_marker_surface!(surf, V, xvi, dt)
        end

        expected = z0 + vz_val * dt * nsteps
        @test all(abs.(Array(surf.topo) .- expected) .< 1e-6)
    end

    @testset "MarkerSurface Extended — Tilted surface" begin
        # Test that a linearly tilted surface is preserved under zero velocity
        xv, yv, zv = make_grid(; nx=8, ny=8, nz=8)
        nx1, ny1 = length(xv), length(yv)

        # Tilted surface: z = 0.3 + 0.2*x + 0.1*y
        z_init = [0.3 + 0.2 * xv[i] + 0.1 * yv[j] for i in 1:nx1, j in 1:ny1]
        surf = init_marker_surface(backend, xv, yv, z_init)

        fill!(surf.vx, 0.0)
        fill!(surf.vy, 0.0)
        fill!(surf.vz, 0.0)

        advect_surface_topo!(surf, 0.1)

        # Interior nodes should be exact (zero velocity → no change)
        topo_cpu = Array(surf.topo)
        interior_err = maximum(abs.(topo_cpu[2:end-1, 2:end-1] .- z_init[2:end-1, 2:end-1]))
        @test interior_err < 1e-12

        # Boundary nodes may have small errors due to neighbor clamping
        # (same limitation as LaMEM's FreeSurfAdvectTopo)
        boundary_err = maximum(abs.(topo_cpu .- z_init))
        @test boundary_err < 0.05
    end


end

@testset "MarkerSurface — Smoothing" begin
    xv, yv, _ = make_grid(; nx=4, ny=4)

    @testset "No smoothing when max_angle=0" begin
        surf = init_marker_surface(backend, xv, yv, 0.5)
        # Add a spike via CPU round-trip
        topo_cpu = Array(surf.topo)
        topo_cpu[3, 3] = 100.0
        copyto!(surf.topo, topo_cpu)
        smooth_surface_max_angle!(surf, 0.0)
        @test Array(surf.topo)[3, 3] ≈ 100.0  # Should not change
    end

    @testset "Smoothing removes steep spikes" begin
        surf = init_marker_surface(backend, xv, yv, 0.5)
        # Add large spike at center
        topo_cpu = Array(surf.topo)
        topo_cpu[3, 3] = 100.0
        copyto!(surf.topo, topo_cpu)
        smooth_surface_max_angle!(surf, 10.0)  # 10 degrees
        # The spike should be smoothed (reduced significantly)
        @test Array(surf.topo)[3, 3] < 100.0
    end

    @testset "Diffusive smoothing" begin
        surf = init_marker_surface(backend, xv, yv, 0.5)
        topo_cpu = Array(surf.topo)
        topo_cpu[3, 3] = 10.0
        copyto!(surf.topo, topo_cpu)
        old_val = 10.0
        smooth_surface_diffusive!(surf, 5)
        # After diffusion, the spike should be reduced
        topo_result = Array(surf.topo)
        @test topo_result[3, 3] < old_val
        # And neighbors should have increased
        @test topo_result[3, 2] > 0.5
    end
    @testset "MarkerSurface Extended — Smoothing preserves flat surfaces" begin
        xv, yv, _ = make_grid(; nx=16, ny=16)

        # Flat surface should not be modified by smoothing
        surf = init_marker_surface(backend, xv, yv, 0.5)
        smooth_surface_max_angle!(surf, 5.0)  # 5 degrees
        @test all(Array(surf.topo) .≈ 0.5)

        # Diffusive smoothing on flat surface: no change
        smooth_surface_diffusive!(surf, 10)
        @test all(Array(surf.topo) .≈ 0.5)
    end

    @testset "MarkerSurface Extended — Smoothing convergence" begin
        xv, yv, _ = make_grid(; nx=16, ny=16)
        nx1, ny1 = length(xv), length(yv)

        # Surface with a sharp spike
        z_init = fill(0.5, nx1, ny1)
        z_init[9, 9] = 5.0  # Large spike

        surf = init_marker_surface(backend, xv, yv, z_init)

        # Repeated diffusive smoothing should converge toward the mean
        for _ in 1:1000
            smooth_surface_diffusive!(surf, 1)
        end

        # After many iterations, the spike should be mostly diffused
        # All values should be close to each other (within the boundary constraints)
        interior = Array(surf.topo)[2:end-1, 2:end-1]
        @test std(interior) < 0.1  # Much less than the original spike
    end

end

# ═════════════════════════════════════════════════════════
# Lightweight mock for RockRatio (avoids JustRelax dependency)
struct MockRockRatio3D{T<:AbstractArray{Float64,3}}
    center::T
    vertex::T
    Vx::T
    Vy::T
    Vz::T
    xy::T
    yz::T
    xz::T
end

function MockRockRatio3D(nx, ny, nz)
    AT = TA(backend)
    return MockRockRatio3D(
        AT(zeros(nx, ny, nz)),
        AT(zeros(nx+1, ny+1, nz+1)),
        AT(zeros(nx+1, ny, nz)),
        AT(zeros(nx, ny+1, nz)),
        AT(zeros(nx, ny, nz+1)),
        AT(zeros(nx+1, ny+1, nz)),
        AT(zeros(nx, ny+1, nz+1)),
        AT(zeros(nx+1, ny, nz+1)),
    )
end

@testset "MarkerSurface — Rock fraction (compute_rock_fraction!)" begin
    xv, yv, zv = make_grid(; nz=4)
    nx, ny, nz_g = length(xv)-1, length(yv)-1, length(zv)-1
    di = (xv[2]-xv[1], yv[2]-yv[1], zv[2]-zv[1])

    @testset "Surface above domain → all rock" begin
        surf = init_marker_surface(backend, xv, yv, 1.5)  # above ztop=1.0
        ϕ = MockRockRatio3D(nx, ny, nz_g)
        compute_rock_fraction!(ϕ, surf, (xv, yv, zv), di)
        @test all(Array(ϕ.center) .≈ 1.0)
    end

    @testset "Surface below domain → all air" begin
        surf = init_marker_surface(backend, xv, yv, -0.5)  # below zbot=0.0
        ϕ = MockRockRatio3D(nx, ny, nz_g)
        compute_rock_fraction!(ϕ, surf, (xv, yv, zv), di)
        @test all(Array(ϕ.center) .≈ 0.0)
    end

    @testset "Surface at mid-height → partial fill" begin
        surf = init_marker_surface(backend, xv, yv, 0.5)
        ϕ = MockRockRatio3D(nx, ny, nz_g)
        compute_rock_fraction!(ϕ, surf, (xv, yv, zv), di)
        center_cpu = Array(ϕ.center)
        for k in 1:nz_g
            if zv[k+1] ≤ 0.5
                # Cells entirely below surface → fully rock
                @test all(center_cpu[:, :, k] .≈ 1.0)
            elseif zv[k] ≥ 0.5
                # Cells entirely above surface → fully air (rock = 0)
                @test all(center_cpu[:, :, k] .≈ 0.0)
            end
        end
    end

    @testset "Staggered nodes are consistent" begin
        surf = init_marker_surface(backend, xv, yv, 0.5)
        ϕ = MockRockRatio3D(nx, ny, nz_g)
        compute_rock_fraction!(ϕ, surf, (xv, yv, zv), di)
        # All staggered values should be in [0, 1]
        @test all(0.0 .≤ Array(ϕ.vertex) .≤ 1.0)
        @test all(0.0 .≤ Array(ϕ.Vx) .≤ 1.0)
        @test all(0.0 .≤ Array(ϕ.Vy) .≤ 1.0)
        @test all(0.0 .≤ Array(ϕ.Vz) .≤ 1.0)
        @test all(0.0 .≤ Array(ϕ.xy) .≤ 1.0)
        @test all(0.0 .≤ Array(ϕ.yz) .≤ 1.0)
        @test all(0.0 .≤ Array(ϕ.xz) .≤ 1.0)
    end

    @testset "MarkerSurface Extended — Rock fraction consistency" begin
        xv, yv, zv = make_grid(; nx=4, ny=4, nz=8)
        nx_g, ny_g, nz_g = 4, 4, length(zv) - 1
        di = (xv[2]-xv[1], yv[2]-yv[1], zv[2]-zv[1])

        # Flat surface at z=0.5 → cells below should be rock, above should be air
        surf = init_marker_surface(backend, xv, yv, 0.5)
        ϕ = MockRockRatio3D_Ext(nx_g, ny_g, nz_g)
        compute_rock_fraction!(ϕ, surf, (xv, yv, zv), di)

        # Check that the rock fraction sums approximately to 50% of total volume
        total_rock = sum(Array(ϕ.center)) / (nx_g * ny_g * nz_g)
        @test abs(total_rock - 0.5) < 0.1  # roughly half-filled

        # Monotonicity: lower cells should have higher rock fraction
        center_cpu = Array(ϕ.center)
        for i in 1:nx_g, j in 1:ny_g
            for k in 1:(nz_g-1)
                @test center_cpu[i, j, k] ≥ center_cpu[i, j, k+1] - 1e-10
            end
        end
    end

end


@testset "MarkerSurface — Triangular prism intersection" begin
    # Verify the prism intersection gives correct rock fractions
    # for a simple geometry

    @testset "Full prism below surface" begin
        cx = (0.0, 1.0, 0.5, 1.0, 0.5)
        cy = (0.0, 0.0, 1.0, 1.0, 0.5)
        cz = (2.0, 2.0, 2.0, 2.0, 2.0)  # surface at z=2, cell is [0,1]
        tri = (1, 2, 5)
        val = _3D._intersect_triangular_prism(cx, cy, cz, tri, 1.0, 0.0, 1.0)
        @test val ≈ 0.25 atol = 1e-10
    end

    @testset "Full prism above surface" begin
        cx = (0.0, 1.0, 0.5, 1.0, 0.5)
        cy = (0.0, 0.0, 1.0, 1.0, 0.5)
        cz = (-1.0, -1.0, -1.0, -1.0, -1.0)  # surface at z=-1, cell is [0,1]
        tri = (1, 2, 5)
        val = _3D._intersect_triangular_prism(cx, cy, cz, tri, 1.0, 0.0, 1.0)
        @test val ≈ 0.0 atol = 1e-10
    end
end
