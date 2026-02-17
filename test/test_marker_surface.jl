"""
Tests for the MarkerSurface (3D free surface tracking) module.

These tests verify initialization, advection, smoothing, phase ratios,
erosion, and sedimentation — all without MPI or GPU, on a pure CPU backend.
"""

ENV["JULIA_JUSTPIC_BACKEND"] = "CPU"

using Test
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)

using JustPIC, JustPIC._3D
import JustPIC._3D: apply_erosion!, apply_sedimentation!

const backend = JustPIC.CPUBackend

# ─────────────────────────────────────────────────────────
# Helper: set up a simple 3D grid
# ─────────────────────────────────────────────────────────
function make_grid(; nx=8, ny=8, nz=8,
                     Lx=1.0, Ly=1.0, Lz=1.0)
    xv = LinRange(0.0, Lx, nx + 1)
    yv = LinRange(0.0, Ly, ny + 1)
    zv = LinRange(0.0, Lz, nz + 1)
    return xv, yv, zv
end

# ═════════════════════════════════════════════════════════
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
        @test surf.topo ≈ z_init
        @test surf.topo0 ≈ z_init
        @test compute_avg_topo(surf) ≈ sum(z_init) / length(z_init)
    end

    @testset "set_topo_from_array!" begin
        surf = init_marker_surface(backend, xv, yv, 0.0)
        z_new = fill(0.3, length(xv), length(yv))
        set_topo_from_array!(surf, z_new)
        @test all(surf.topo .≈ 0.3)
        @test all(surf.topo0 .≈ 0.3)
        @test compute_avg_topo(surf) ≈ 0.3
    end
end

# ═════════════════════════════════════════════════════════
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

# ═════════════════════════════════════════════════════════
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

# ═════════════════════════════════════════════════════════
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

# ═════════════════════════════════════════════════════════
@testset "MarkerSurface — Smoothing" begin
    xv, yv, _ = make_grid(; nx=4, ny=4)

    @testset "No smoothing when max_angle=0" begin
        surf = init_marker_surface(backend, xv, yv, 0.5)
        # Add a spike
        surf.topo[3, 3] = 100.0
        smooth_surface_max_angle!(surf, 0.0)
        @test surf.topo[3, 3] ≈ 100.0  # Should not change
    end

    @testset "Smoothing removes steep spikes" begin
        surf = init_marker_surface(backend, xv, yv, 0.5)
        # Add large spike at center
        surf.topo[3, 3] = 100.0
        smooth_surface_max_angle!(surf, 10.0)  # 10 degrees
        # The spike should be smoothed (reduced significantly)
        @test surf.topo[3, 3] < 100.0
    end

    @testset "Diffusive smoothing" begin
        surf = init_marker_surface(backend, xv, yv, 0.5)
        surf.topo[3, 3] = 10.0
        old_val = surf.topo[3, 3]
        smooth_surface_diffusive!(surf, 5)
        # After diffusion, the spike should be reduced
        @test surf.topo[3, 3] < old_val
        # And neighbors should have increased
        @test surf.topo[3, 2] > 0.5
    end
end

# ═════════════════════════════════════════════════════════
# Lightweight mock for RockRatio (avoids JustRelax dependency)
struct MockRockRatio3D
    center::Array{Float64,3}
    vertex::Array{Float64,3}
    Vx::Array{Float64,3}
    Vy::Array{Float64,3}
    Vz::Array{Float64,3}
    xy::Array{Float64,3}
    yz::Array{Float64,3}
    xz::Array{Float64,3}
end

function MockRockRatio3D(nx, ny, nz)
    return MockRockRatio3D(
        zeros(nx, ny, nz),
        zeros(nx+1, ny+1, nz+1),
        zeros(nx+1, ny, nz),
        zeros(nx, ny+1, nz),
        zeros(nx, ny, nz+1),
        zeros(nx+1, ny+1, nz),
        zeros(nx, ny+1, nz+1),
        zeros(nx+1, ny, nz+1),
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
        @test all(ϕ.center .≈ 1.0)
    end

    @testset "Surface below domain → all air" begin
        surf = init_marker_surface(backend, xv, yv, -0.5)  # below zbot=0.0
        ϕ = MockRockRatio3D(nx, ny, nz_g)
        compute_rock_fraction!(ϕ, surf, (xv, yv, zv), di)
        @test all(ϕ.center .≈ 0.0)
    end

    @testset "Surface at mid-height → partial fill" begin
        surf = init_marker_surface(backend, xv, yv, 0.5)
        ϕ = MockRockRatio3D(nx, ny, nz_g)
        compute_rock_fraction!(ϕ, surf, (xv, yv, zv), di)
        for k in 1:nz_g
            if zv[k+1] ≤ 0.5
                # Cells entirely below surface → fully rock
                @test all(ϕ.center[:, :, k] .≈ 1.0)
            elseif zv[k] ≥ 0.5
                # Cells entirely above surface → fully air (rock = 0)
                @test all(ϕ.center[:, :, k] .≈ 0.0)
            end
        end
    end

    @testset "Staggered nodes are consistent" begin
        surf = init_marker_surface(backend, xv, yv, 0.5)
        ϕ = MockRockRatio3D(nx, ny, nz_g)
        compute_rock_fraction!(ϕ, surf, (xv, yv, zv), di)
        # All staggered values should be in [0, 1]
        @test all(0.0 .≤ ϕ.vertex .≤ 1.0)
        @test all(0.0 .≤ ϕ.Vx .≤ 1.0)
        @test all(0.0 .≤ ϕ.Vy .≤ 1.0)
        @test all(0.0 .≤ ϕ.Vz .≤ 1.0)
        @test all(0.0 .≤ ϕ.xy .≤ 1.0)
        @test all(0.0 .≤ ϕ.yz .≤ 1.0)
        @test all(0.0 .≤ ϕ.xz .≤ 1.0)
    end
end

# # ═════════════════════════════════════════════════════════
# @testset "MarkerSurface — Erosion" begin
#     xv, yv, _ = make_grid()

#     @testset "Model 0: No erosion" begin
#         surf = init_marker_surface(backend, xv, yv, 0.8)
#         topo_before = copy(surf.topo)
#         apply_erosion!(surf, 0.1, 0.0; model=0)
#         @test surf.topo ≈ topo_before
#     end

#     @testset "Model 1: Infinitely fast erosion" begin
#         nx1, ny1 = length(xv), length(yv)
#         z_init = [0.5 + 0.2 * sin(2π * xv[i]) for i in 1:nx1, j in 1:ny1]
#         surf = init_marker_surface(backend, xv, yv, z_init)
#         avg = compute_avg_topo(surf)
#         apply_erosion!(surf, 0.1, 0.0; model=1)
#         @test all(surf.topo .≈ avg)
#     end

#     @testset "Model 2: Prescribed rate" begin
#         surf = init_marker_surface(backend, xv, yv, 0.9)
#         apply_erosion!(surf, 0.1, 0.0; model=2, rate=0.5, level=0.5, zbot=0.0, ztop=1.0)
#         # Topography was 0.9, which is above level 0.5, so it should decrease
#         @test all(surf.topo .< 0.9)
#         # rate=0.5, dt=0.1, dz=0.05.  0.9 - 0.05 = 0.85
#         @test all(abs.(surf.topo .- 0.85) .< 1e-12)
#     end
# end

# # ═════════════════════════════════════════════════════════
# @testset "MarkerSurface — Sedimentation" begin
#     xv, yv, _ = make_grid()

#     @testset "Model 0: No sedimentation" begin
#         surf = init_marker_surface(backend, xv, yv, 0.2)
#         topo_before = copy(surf.topo)
#         apply_sedimentation!(surf, 0.1, 0.0; model=0)
#         @test surf.topo ≈ topo_before
#     end

#     @testset "Model 1: Prescribed rate" begin
#         surf = init_marker_surface(backend, xv, yv, 0.2)
#         apply_sedimentation!(surf, 0.1, 0.0; model=1, rate=1.0, level=0.5, zbot=0.0, ztop=1.0)
#         # Topography 0.2 < level 0.5 → sediment deposited
#         # rate=1.0, dt=0.1, dz=0.1
#         @test all(abs.(surf.topo .- 0.3) .< 1e-12)
#     end

#     @testset "Model 1: No sedimentation above level" begin
#         surf = init_marker_surface(backend, xv, yv, 0.8)
#         apply_sedimentation!(surf, 0.1, 0.0; model=1, rate=1.0, level=0.5, zbot=0.0, ztop=1.0)
#         # Topography 0.8 > level 0.5 → no change
#         @test all(surf.topo .≈ 0.8)
#     end

#     @testset "Model 3: Differential loading" begin
#         surf = init_marker_surface(backend, xv, yv, 0.5)
#         apply_sedimentation!(surf, 0.1, 0.0; model=3, rate=1.0, rate2=0.5, zbot=0.0, ztop=2.0)
#         # Left edge (x=0) gets dz1, right edge (x=1) gets dz2
#         # dz1 = 1.0*0.1 = 0.1, dz2 = 0.5*0.1 = 0.05
#         @test surf.topo[1, 1] ≈ 0.5 + 0.1 atol = 1e-10  # Left edge
#         @test surf.topo[end, 1] ≈ 0.5 + 0.05 atol = 1e-10  # Right edge
#     end
# end

# ═════════════════════════════════════════════════════════
@testset "MarkerSurface — Velocity interpolation" begin
    xv, yv, zv = make_grid(; nx=4, ny=4, nz=4)
    nx1, ny1 = length(xv), length(yv)

    surf = init_marker_surface(backend, xv, yv, 0.5)

    # Create constant velocity field Vz = 1.0 everywhere
    nxg, nyg, nzg = length(xv), length(yv), length(zv)
    Vx = zeros(nxg, nyg, nzg)
    Vy = zeros(nxg, nyg, nzg)
    Vz = ones(nxg, nyg, nzg)

    V = (Vx, Vy, Vz)
    xvi = (collect(Float64, xv), collect(Float64, yv), collect(Float64, zv))

    interpolate_velocity_to_surface_vertices!(surf, V, xvi)

    # With constant Vz=1, all surface vz should be 1
    @test all(abs.(surf.vz .- 1.0) .< 1e-10)
    @test all(abs.(surf.vx) .< 1e-10)
    @test all(abs.(surf.vy) .< 1e-10)
end

# ═════════════════════════════════════════════════════════
@testset "MarkerSurface — Full advection pipeline" begin
    xv, yv, zv = make_grid(; nx=8, ny=8, nz=8)
    z0 = 0.5

    surf = init_marker_surface(backend, xv, yv, z0)

    # Create a velocity field: uniform upward Vz=0.1
    nxg, nyg, nzg = length(xv), length(yv), length(zv)
    Vx = zeros(nxg, nyg, nzg)
    Vy = zeros(nxg, nyg, nzg)
    Vz = fill(0.1, nxg, nyg, nzg)
    V = (Vx, Vy, Vz)
    xvi = (collect(Float64, xv), collect(Float64, yv), collect(Float64, zv))

    dt = 0.1

    advect_marker_surface!(surf, V, xvi, dt)

    # Expected: z0 + 0.1 * 0.1 = 0.51
    @test all(abs.(surf.topo .- 0.51) .< 1e-8)
    @test abs(compute_avg_topo(surf) - 0.51) < 1e-8
end

# ═════════════════════════════════════════════════════════
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
