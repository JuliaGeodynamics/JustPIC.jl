"""
Extended tests for the MarkerSurface module.
Tests edge cases, convergence, and more complex scenarios.
"""

ENV["JULIA_JUSTPIC_BACKEND"] = "CPU"

using Test
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)

using JustPIC, JustPIC._3D
import JustPIC._3D: apply_erosion!, apply_sedimentation!
using LinearAlgebra
using Statistics: std

const backend = JustPIC.CPUBackend

function make_grid(; nx=8, ny=8, nz=8,
                     Lx=1.0, Ly=1.0, Lz=1.0)
    xv = LinRange(0.0, Lx, nx + 1)
    yv = LinRange(0.0, Ly, ny + 1)
    zv = LinRange(0.0, Lz, nz + 1)
    return xv, yv, zv
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
    max_change = maximum(abs.(surf.topo .- z_init))
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
    # Flat topography with zero velocity → should remain flat
    @test all(abs.(surf.topo .- z0) .< 1e-10)
end

@testset "MarkerSurface Extended — Smoothing preserves flat surfaces" begin
    xv, yv, _ = make_grid(; nx=16, ny=16)

    # Flat surface should not be modified by smoothing
    surf = init_marker_surface(backend, xv, yv, 0.5)
    smooth_surface_max_angle!(surf, 5.0)  # 5 degrees
    @test all(surf.topo .≈ 0.5)

    # Diffusive smoothing on flat surface: no change
    smooth_surface_diffusive!(surf, 10)
    @test all(surf.topo .≈ 0.5)
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
    interior = surf.topo[2:end-1, 2:end-1]
    @test std(interior) < 0.1  # Much less than the original spike
end

# Lightweight mock for RockRatio (avoids JustRelax dependency)
struct MockRockRatio3D_Ext
    center::Array{Float64,3}
    vertex::Array{Float64,3}
    Vx::Array{Float64,3}
    Vy::Array{Float64,3}
    Vz::Array{Float64,3}
    xy::Array{Float64,3}
    yz::Array{Float64,3}
    xz::Array{Float64,3}
end

function MockRockRatio3D_Ext(nx, ny, nz)
    return MockRockRatio3D_Ext(
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

@testset "MarkerSurface Extended — Rock fraction consistency" begin
    xv, yv, zv = make_grid(; nx=4, ny=4, nz=8)
    nx_g, ny_g, nz_g = 4, 4, length(zv) - 1
    di = (xv[2]-xv[1], yv[2]-yv[1], zv[2]-zv[1])

    # Flat surface at z=0.5 → cells below should be rock, above should be air
    surf = init_marker_surface(backend, xv, yv, 0.5)
    ϕ = MockRockRatio3D_Ext(nx_g, ny_g, nz_g)
    compute_rock_fraction!(ϕ, surf, (xv, yv, zv), di)

    # Check that the rock fraction sums approximately to 50% of total volume
    total_rock = sum(ϕ.center) / (nx_g * ny_g * nz_g)
    @test abs(total_rock - 0.5) < 0.1  # roughly half-filled

    # Monotonicity: lower cells should have higher rock fraction
    for i in 1:nx_g, j in 1:ny_g
        for k in 1:(nz_g-1)
            @test ϕ.center[i, j, k] ≥ ϕ.center[i, j, k+1] - 1e-10
        end
    end
end

@testset "MarkerSurface Extended — Multiple timestep advection" begin
    xv, yv, zv = make_grid(; nx=8, ny=8, nz=16)
    z0 = 0.3
    vz_val = 0.1
    dt = 0.01
    nsteps = 10

    surf = init_marker_surface(backend, xv, yv, z0)

    nxg, nyg, nzg = length(xv), length(yv), length(zv)
    Vx = zeros(nxg, nyg, nzg)
    Vy = zeros(nxg, nyg, nzg)
    Vz = fill(vz_val, nxg, nyg, nzg)
    V = (Vx, Vy, Vz)
    xvi = (collect(Float64, xv), collect(Float64, yv), collect(Float64, zv))

    for _ in 1:nsteps
        advect_marker_surface!(surf, V, xvi, dt)
    end

    expected = z0 + vz_val * dt * nsteps
    @test all(abs.(surf.topo .- expected) .< 1e-6)
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
    interior_err = maximum(abs.(surf.topo[2:end-1, 2:end-1] .- z_init[2:end-1, 2:end-1]))
    @test interior_err < 1e-12

    # Boundary nodes may have small errors due to neighbor clamping
    # (same limitation as LaMEM's FreeSurfAdvectTopo)
    boundary_err = maximum(abs.(surf.topo .- z_init))
    @test boundary_err < 0.05
end
