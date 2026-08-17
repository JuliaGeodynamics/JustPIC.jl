const BACKEND_NAME = get(ENV, "JULIA_JUSTPIC_BACKEND", "CPU")

@static if BACKEND_NAME == "AMDGPU"
    using AMDGPU
    AMDGPU.allowscalar(true)
elseif BACKEND_NAME == "CUDA"
    using CUDA
    CUDA.allowscalar(true)
elseif BACKEND_NAME == "Metal"
    using Metal
    Metal.allowscalar(true)
end

using JustPIC, CellArrays, Test, LinearAlgebra
using ImplicitGlobalGrid
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

const FT = if BACKEND_NAME == "Metal" || get(ENV, "JULIA_JUSTPIC_PRECISION", "") == "Float32"
    Float32
else
    Float64
end

function make_grid(;
        nx = 8, ny = 8, nz = 8,
        Lx = FT(1), Ly = FT(1), Lz = FT(1)
    )
    xv = LinRange(zero(FT), Lx, nx + 1)
    yv = LinRange(zero(FT), Ly, ny + 1)
    zv = LinRange(zero(FT), Lz, nz + 1)
    return xv, yv, zv
end

function extend_centers(xv)
    xc = [(xv[i] + xv[i + 1]) / 2 for i in 1:(length(xv) - 1)]
    return [2 * xc[1] - xc[2]; xc; 2 * xc[end] - xc[end - 1]]
end

function staggered_velocity_grid(xv, yv, zv)
    return (
        (collect(xv), extend_centers(yv), extend_centers(zv)),
        (extend_centers(xv), collect(yv), extend_centers(zv)),
        (extend_centers(xv), extend_centers(yv), collect(zv)),
    )
end

device_grid(grid) = ntuple(d -> TA(backend).(grid[d]), Val(3))

function sample_velocity(f, grid_vxi)
    return ntuple(Val(3)) do d
        x, y, z = Array.(grid_vxi[d])
        TA(backend)([f(d, x_, y_, z_) for x_ in x, y_ in y, z_ in z])
    end
end

@testset "MarkerSurface" begin
    @test !isdefined(JustPIC, :semilagrangian_advect_surface!)
    @test !isdefined(JustPIC, :smooth_surface_diffusive!)

    @testset "MarkerSurface — Initialization 3D" begin
        xv, yv, zv = make_grid()

        @testset "Flat surface (scalar elevation)" begin
            surf = init_marker_surface(backend, xv, yv, FT(0.5))
            @test size(surf.topo) == (length(xv), length(yv))
            @test all(Array(surf.topo) .≈ FT(0.5))
            @test all(Array(surf.topo0) .≈ FT(0.5))
            @test compute_avg_topo(surf) ≈ FT(0.5)
        end

        @testset "Precision follows inputs" begin
            for T in (Float32, Float64)
                BACKEND_NAME == "Metal" && T === Float64 && continue
                x = LinRange(zero(T), one(T), 5)
                y = LinRange(zero(T), one(T), 4)
                elevation = T(0.25)
                surf = init_marker_surface(backend, x, y, elevation)
                @test eltype(surf.topo) === T
                @test eltype(surf.xv) === T
                @test eltype(surf.yv) === T

                z = TA(backend)([T(0.1) + x_ + y_ for x_ in x, y_ in y])
                array_surf = init_marker_surface(backend, x, y, z)
                @test eltype(array_surf.topo) === T
                @test Array(array_surf.topo) ≈ Array(z)
            end

            x = LinRange(zero(FT), one(FT), 5)
            y = LinRange(zero(FT), one(FT), 4)
            surf = init_marker_surface(backend, x, y, FT(0.5))
            @test eltype(surf.topo) === FT
            @test size(surf.advection_valid) == size(surf.topo)
            @test size(surf.smoothing_cell_topo) == size(surf.topo) .- 1
            @test size(surf.smoothing_steep) == size(surf.topo) .- 1
            @test size(surf.z_ownership) == size(surf.topo)

            surf_copy = copy(surf)
            @test surf_copy !== surf
            @test Array(surf_copy.topo) == Array(surf.topo)
            @test eltype(surf_copy.advection_valid) === Bool

            cpu_surface = Array(surf)
            @test eltype(cpu_surface.topo) === FT
            @test eltype(cpu_surface.advection_valid) === Bool
            @test eltype(cpu_surface.smoothing_steep) === Bool

            if BACKEND_NAME != "Metal"
                promoted = init_marker_surface(backend, Float32[0, 1], Float32[0, 1], 0.5)
                @test eltype(promoted.topo) === Float64
                @test eltype(promoted.xv) === Float64
                @test eltype(promoted.yv) === Float64
            end
        end

        @testset "Invalid initialization fails explicitly" begin
            @test_throws ArgumentError init_marker_surface(backend, FT[0, 0], FT[0, 1], FT(0))
            @test_throws ArgumentError init_marker_surface(backend, FT[0, 1], FT[0, FT(NaN)], FT(0))
            @test_throws DimensionMismatch init_marker_surface(backend, xv, yv, zeros(FT, 2, 2))
        end

        @testset "Variable initial topography" begin
            nx1, ny1 = length(xv), length(yv)
            z_init = [
                FT(0.4) + FT(0.1) * sin(FT(2π) * xv[i]) * cos(FT(2π) * yv[j])
                    for i in 1:nx1, j in 1:ny1
            ]
            surf = init_marker_surface(backend, xv, yv, z_init)
            @test Array(surf.topo) ≈ z_init
            @test Array(surf.topo0) ≈ z_init
            @test compute_avg_topo(surf) ≈ sum(z_init) / length(z_init)
        end

        @testset "set_topo_from_array!" begin
            surf = init_marker_surface(backend, xv, yv, zero(FT))
            AT = TA(backend)
            z_new = AT(fill(FT(0.3), length(xv), length(yv)))
            set_topo_from_array!(surf, z_new)
            @test all(Array(surf.topo) .≈ FT(0.3))
            @test all(Array(surf.topo0) .≈ FT(0.3))
            @test compute_avg_topo(surf) ≈ FT(0.3)
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
            ok, zp = JustPIC._interpolate_triangle(cx, cy, cz, tri, xp, yp)
            @test ok == true
            # At centroid, barycentric coords are (1/3, 1/3, 1/3)
            @test zp ≈ (1.0 + 2.0 + 3.0) / 3 atol = 1.0e-10
        end

        @testset "Point outside triangle" begin
            cx = (0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            cy = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            cz = (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            tri = (1, 2, 3)
            ok, _ = JustPIC._interpolate_triangle(cx, cy, cz, tri, -1.0, -1.0)
            @test ok == false
        end
    end

    @testset "MarkerSurface - Advection" begin
        @testset "MarkerSurface — Physical boundary ghosts clamp fields 3D" begin
            field = reshape(FT.(1:9), 3, 3)
            @test JustPIC._ghost_coord(FT[0, 1, 3], 0, 3) == FT(-1)
            @test JustPIC._ghost_field(field, 0, 0, 3, 3, false, false) == field[1, 1]
            @test JustPIC._ghost_field(field, 4, 4, 3, 3, false, false) == field[3, 3]
            @test JustPIC._ghost_field(field, 0, 2, 3, 3, true, false) == field[2, 2]
            @test JustPIC._ghost_field(field, 2, 4, 3, 3, false, true) == field[2, 2]

            periodic_field = reshape(FT.(1:16), 4, 4)
            @test JustPIC._ghost_field(periodic_field, 0, 2, 4, 4, true, false) == periodic_field[3, 2]
            @test JustPIC._ghost_field(periodic_field, 5, 2, 4, 4, true, false) == periodic_field[2, 2]
        end

        @testset "MarkerSurface — Advection (zero velocity)" begin
            xv, yv, zv = make_grid()
            z0 = FT(0.5)

            surf = init_marker_surface(backend, xv, yv, z0)
            dt = FT(0.01)

            # Zero velocity field → topography should not change
            nx1, ny1 = length(xv), length(yv)
            fill!(surf.vx, zero(FT))
            fill!(surf.vy, zero(FT))
            fill!(surf.vz, zero(FT))

            advect_surface_topo!(surf, dt)
            @test all(abs.(Array(surf.topo) .- z0) .≤ eps(FT) * 8)
        end

        @testset "MarkerSurface — Advection (uniform vertical velocity)" begin
            xv, yv, zv = make_grid()
            z0 = FT(0.5)
            vz_val = FT(0.1)
            dt = FT(0.1)

            surf = init_marker_surface(backend, xv, yv, z0)

            # Set uniform vertical velocity, zero horizontal
            fill!(surf.vx, zero(FT))
            fill!(surf.vy, zero(FT))
            fill!(surf.vz, FT(vz_val))

            advect_surface_topo!(surf, dt)

            # Expected: z0 + vz*dt = 0.5 + 0.01 = 0.51
            expected = z0 + vz_val * dt
            @test all(abs.(Array(surf.topo) .- expected) .≤ eps(FT) * 8)
        end

        @testset "MarkerSurface — Periodic seam" begin
            xv, yv, _ = make_grid()
            topo = [FT(0.5) + FT(0.05) * cos(FT(2π) * x) * cos(FT(2π) * y) for x in xv, y in yv]
            topo[end, :] .= topo[1, :]
            topo[:, end] .= topo[:, 1]
            surf = init_marker_surface(backend, xv, yv, topo; periodic_1 = true, periodic_2 = true)
            fill!(surf.vx, zero(FT))
            fill!(surf.vy, zero(FT))
            fill!(surf.vz, FT(0.1))

            advect_surface_topo!(surf, FT(0.1))
            result = Array(surf.topo)
            @test result[end, :] ≈ result[1, :]
            @test result[:, end] ≈ result[:, 1]
            @test compute_avg_topo(surf) ≈ sum(result[1:(end - 1), 1:(end - 1)]) / (length(xv) - 1)^2
        end

        @testset "MarkerSurface — Periodic horizontal transport" begin
            xv, yv, zv = make_grid(; nx = 32, ny = 8, nz = 8)
            u, dt = FT(0.1), FT(0.02)
            topo = [FT(0.5) + FT(0.05) * cos(FT(2π) * x) * cos(FT(2π) * y) for x in xv, y in yv]
            topo[end, :] .= topo[1, :]
            surf = init_marker_surface(backend, xv, yv, topo; periodic_1 = true)
            grid_vxi = device_grid(staggered_velocity_grid(xv, yv, zv))
            V = sample_velocity(grid_vxi) do d, x, y, z
                d == 1 ? u : zero(FT)
            end

            advect_marker_surface!(surf, V, grid_vxi, dt; max_slope_angle = zero(FT))

            result = Array(surf.topo)
            expected = [FT(0.5) + FT(0.05) * cos(FT(2π) * (x - u * dt)) * cos(FT(2π) * y) for x in xv, y in yv]
            @test result[1, :] ≈ expected[1, :] atol = FT(2.0e-4)
            @test result[end, :] ≈ expected[end, :] atol = FT(2.0e-4)
        end

        @testset "MarkerSurface — Staggered affine interpolation 3D" begin
            for (xv, yv, zv) in (
                    make_grid(; nx = 4, ny = 5, nz = 4),
                    (FT[0, 0.12, 0.38, 0.7, 1], FT[0, 0.25, 0.55, 1], FT[0, 0.18, 0.48, 0.76, 1]),
                )
                topo = [FT(0.2) + FT(0.1) * x + FT(0.05) * y for x in xv, y in yv]
                surf = init_marker_surface(backend, xv, yv, topo)
                grid_vxi = device_grid(staggered_velocity_grid(xv, yv, zv))
                V = sample_velocity(grid_vxi) do d, x, y, z
                    d == 1 && return FT(0.2) + FT(1.3) * x - FT(0.4) * y + FT(0.7) * z
                    d == 2 && return -FT(0.1) + FT(0.2) * x + FT(0.9) * y - FT(0.3) * z
                    return FT(0.4) - FT(0.5) * x + FT(0.6) * y + FT(1.1) * z
                end

                interpolate_velocity_to_surface_vertices!(surf, V, grid_vxi)
                atol = FT === Float32 ? FT(3.0e-6) : FT(3.0e-13)
                expected = ntuple(Val(3)) do d
                    [
                        d == 1 ? FT(0.2) + FT(1.3) * x - FT(0.4) * y + FT(0.7) * topo[i, j] :
                            d == 2 ? -FT(0.1) + FT(0.2) * x + FT(0.9) * y - FT(0.3) * topo[i, j] :
                            FT(0.4) - FT(0.5) * x + FT(0.6) * y + FT(1.1) * topo[i, j]
                            for (i, x) in pairs(xv), (j, y) in pairs(yv)
                    ]
                end
                @test Array(surf.vx) ≈ expected[1] atol = atol rtol = atol
                @test Array(surf.vy) ≈ expected[2] atol = atol rtol = atol
                @test Array(surf.vz) ≈ expected[3] atol = atol rtol = atol

                # The collocated vertex grid has the wrong dimensions for every
                # actual staggered component. Accepting it would recreate P0.
                @test_throws ArgumentError interpolate_velocity_to_surface_vertices!(
                    surf, V, (xv, yv, zv),
                )
            end

            coords = range(FT(-1); step = FT(0.125), length = 17)
            coords_vector = collect(coords)
            boundary = coords[8]
            for value in (
                    coords[1] - step(coords), coords[1], prevfloat(boundary),
                    boundary, nextfloat(boundary), coords[end], coords[end] + step(coords),
                )
                @test JustPIC._find_cell_1d(coords, value) ==
                    JustPIC._find_cell_1d(coords_vector, value)
            end

            xv, yv, zv = make_grid(; nx = 5, ny = 4, nz = 6)
            vector_grid = staggered_velocity_grid(xv, yv, zv)
            range_grid = ntuple(Val(3)) do d
                map(vector_grid[d]) do x
                    range(first(x); step = x[2] - x[1], length = length(x))
                end
            end
            topo = [FT(0.25) + FT(0.1) * x - FT(0.05) * y for x in xv, y in yv]
            range_surf = init_marker_surface(backend, xv, yv, topo)
            vector_surf = init_marker_surface(backend, xv, yv, topo)
            V = sample_velocity(range_grid) do d, x, y, z
                FT(d) + FT(0.2) * x - FT(0.3) * y + FT(0.4) * z
            end
            interpolate_velocity_to_surface_vertices!(range_surf, V, range_grid)
            interpolate_velocity_to_surface_vertices!(vector_surf, V, device_grid(vector_grid))
            atol = FT === Float32 ? FT(3.0e-6) : FT(3.0e-13)
            @test Array(range_surf.vx) ≈ Array(vector_surf.vx) atol = atol rtol = atol
            @test Array(range_surf.vy) ≈ Array(vector_surf.vy) atol = atol rtol = atol
            @test Array(range_surf.vz) ≈ Array(vector_surf.vz) atol = atol rtol = atol
        end

        @testset "MarkerSurface — Full advection pipeline" begin
            xv, yv, zv = make_grid(; nx = 8, ny = 8, nz = 8)
            z0 = FT(0.5)

            surf = init_marker_surface(backend, xv, yv, z0)

            # Create a velocity field on the actual staggered layout.
            grid_vxi = device_grid(staggered_velocity_grid(xv, yv, zv))
            V = sample_velocity(grid_vxi) do d, x, y, z
                d == 3 ? FT(0.1) : zero(FT)
            end

            dt = 0.1

            advect_marker_surface!(surf, V, grid_vxi, dt)

            # Expected: z0 + 0.1 * 0.1 = 0.51
            @test all(abs.(Array(surf.topo) .- FT(0.51)) .≤ eps(FT) * 8)
            @test abs(compute_avg_topo(surf) - FT(0.51)) ≤ eps(FT) * 8
        end

        @testset "MarkerSurface — Interior planar transport is physically consistent 3D" begin
            xv = FT[0, 0.12, 0.38, 0.7, 1]
            yv = FT[0, 0.3, 0.65, 1]
            zv = FT[0, 0.2, 0.5, 0.8, 1]
            a, bx, by = FT(0.3), FT(0.16), FT(-0.11)
            ux, uy, uz, dt = FT(0.08), FT(-0.05), FT(0.04), FT(0.1)
            h0 = [a + bx * x + by * y for x in xv, y in yv]
            surf = init_marker_surface(backend, xv, yv, h0)
            grid_vxi = device_grid(staggered_velocity_grid(xv, yv, zv))
            V = sample_velocity(grid_vxi) do d, x, y, z
                d == 1 ? ux : d == 2 ? uy : uz
            end

            advect_marker_surface!(surf, V, grid_vxi, dt; max_slope_angle = zero(FT))

            # h_t + uₓ h_x + u_y h_y = u_z: away from physical boundaries, an
            # Eulerian planar surface changes by (u_z - uₓ bₓ - u_y b_y)dt.
            # Boundary nodes use the required clamped-field stencil.
            expected = h0 .+ (uz - ux * bx - uy * by) * dt
            atol = FT === Float32 ? FT(3.0e-6) : FT(3.0e-13)
            @test Array(surf.topo)[2:(end - 1), 2:(end - 1)] ≈ expected[2:(end - 1), 2:(end - 1)] atol = atol rtol = atol
        end

        @testset "MarkerSurface — Invalid geometry fails fast 3D" begin
            xv, yv, zv = make_grid(; nx = 4, ny = 4, nz = 4)
            surf = init_marker_surface(backend, xv, yv, FT(0.5))
            vx = [-FT(2) * x for x in xv, _ in yv]
            copyto!(surf.vx, TA(backend)(vx))
            fill!(surf.vy, zero(FT))
            fill!(surf.vz, zero(FT))
            before = Array(surf.topo)
            @test_throws "outside its deformed-grid stencil" advect_surface_topo!(surf, one(FT))
            @test Array(surf.topo) == before

            outside = init_marker_surface(backend, xv, yv, FT(1.1))
            grid_vxi = device_grid(staggered_velocity_grid(xv, yv, zv))
            V = sample_velocity(grid_vxi) do d, x, y, z
                zero(FT)
            end
            @test_throws "outside the vertical velocity grid" interpolate_velocity_to_surface_vertices!(outside, V, grid_vxi)

            nonfinite = init_marker_surface(backend, xv, yv, FT(0.5))
            set_topo_from_array!(nonfinite, TA(backend)(fill(FT(NaN), length(xv), length(yv))))
            @test_throws "topography must be finite" advect_surface_topo!(nonfinite, one(FT))
        end

        @testset "MarkerSurface Extended — Advection convergence" begin
            # Test that uniform vertical uplift is exact regardless of resolution
            for nx in [4, 8, 16, 32]
                xv, yv, zv = make_grid(; nx = nx, ny = nx, nz = nx)
                z0 = FT(0.5)
                vz_val = FT(0.2)
                dt = FT(0.05)
                surf = init_marker_surface(backend, xv, yv, z0)
                fill!(surf.vx, zero(FT))
                fill!(surf.vy, zero(FT))
                fill!(surf.vz, FT(vz_val))
                advect_surface_topo!(surf, dt)
                expected = z0 + vz_val * dt
                err = maximum(abs.(Array(surf.topo) .- expected))
                @test err ≤ eps(FT) * 8
            end
        end

        @testset "MarkerSurface Extended — Horizontal advection" begin
            # When there's only horizontal velocity (vx), the topography shape
            # should translate horizontally. Test with a sinusoidal surface.
            xv, yv, zv = make_grid(; nx = 32, ny = 4, nz = 4)
            nx1, ny1 = length(xv), length(yv)

            z_init = [FT(0.5) + FT(0.1) * sin(FT(2π) * xv[i]) for i in 1:nx1, j in 1:ny1]
            surf = init_marker_surface(backend, xv, yv, z_init)

            # Pure horizontal advection at small vx
            fill!(surf.vx, FT(0.01))
            fill!(surf.vy, zero(FT))
            fill!(surf.vz, zero(FT))

            dt = 0.001  # Small dt for accuracy
            advect_surface_topo!(surf, dt)

            # The surface should have shifted slightly; magnitude of change should be small
            max_change = maximum(abs.(Array(surf.topo) .- z_init))
            @test max_change > 0  # Something should change
            @test max_change < 0.01  # But not by much with small dt
        end

        @testset "MarkerSurface Extended — Multiple timestep advection" begin
            xv, yv, zv = make_grid(; nx = 8, ny = 8, nz = 16)
            z0 = FT(0.3)
            vz_val = FT(0.1)
            dt = FT(0.01)
            nsteps = 10

            surf = init_marker_surface(backend, xv, yv, z0)

            grid_vxi = device_grid(staggered_velocity_grid(xv, yv, zv))
            V = sample_velocity(grid_vxi) do d, x, y, z
                d == 3 ? FT(vz_val) : zero(FT)
            end

            for _ in 1:nsteps
                advect_marker_surface!(surf, V, grid_vxi, dt)
            end

            expected = z0 + vz_val * dt * nsteps
            @test all(abs.(Array(surf.topo) .- expected) .< 1.0e-6)
        end

        @testset "MarkerSurface Extended — Tilted surface" begin
            # Test that a linearly tilted surface is preserved under zero velocity
            xv, yv, zv = make_grid(; nx = 8, ny = 8, nz = 8)
            nx1, ny1 = length(xv), length(yv)

            # Tilted surface: z = 0.3 + 0.2*x + 0.1*y
            z_init = [FT(0.3) + FT(0.2) * xv[i] + FT(0.1) * yv[j] for i in 1:nx1, j in 1:ny1]
            surf = init_marker_surface(backend, xv, yv, z_init)

            fill!(surf.vx, zero(FT))
            fill!(surf.vy, zero(FT))
            fill!(surf.vz, zero(FT))

            advect_surface_topo!(surf, 0.1)

            # Interior nodes should be exact (zero velocity → no change)
            topo_cpu = Array(surf.topo)
            interior_err = maximum(abs.(topo_cpu[2:(end - 1), 2:(end - 1)] .- z_init[2:(end - 1), 2:(end - 1)]))
            @test interior_err ≤ eps(FT) * 8

            # Boundary nodes may have small errors due to neighbor clamping
            # (same limitation as LaMEM's FreeSurfAdvectTopo)
            boundary_err = maximum(abs.(topo_cpu .- z_init))
            @test boundary_err < 0.05
        end


    end

    @testset "MarkerSurface — Smoothing" begin
        xv, yv, _ = make_grid(; nx = 4, ny = 4)

        @testset "No smoothing when max_angle=0" begin
            surf = init_marker_surface(backend, xv, yv, FT(0.5))
            # Add a spike via CPU round-trip
            topo_cpu = Array(surf.topo)
            topo_cpu[3, 3] = 100.0
            copyto!(surf.topo, topo_cpu)
            smooth_surface_max_angle!(surf, 0.0)
            @test Array(surf.topo)[3, 3] ≈ 100.0  # Should not change
        end

        @testset "Smoothing removes steep spikes" begin
            surf = init_marker_surface(backend, xv, yv, FT(0.5))
            # Add large spike at center
            topo_cpu = Array(surf.topo)
            topo_cpu[3, 3] = 100.0
            copyto!(surf.topo, topo_cpu)
            smooth_surface_max_angle!(surf, 10.0)  # 10 degrees
            # The spike should be smoothed (reduced significantly)
            topo_result = Array(surf.topo)
            @test topo_result[3, 3] < 100.0
            @test topo_result[1, 1] == FT(0.5)
            @test all(topo_result[[1, 5], :] .== FT(0.5))
            @test all(topo_result[:, [1, 5]] .== FT(0.5))
        end

        @testset "MarkerSurface Extended — Smoothing preserves flat surfaces" begin
            xv, yv, _ = make_grid(; nx = 16, ny = 16)

            # Flat surface should not be modified by smoothing
            surf = init_marker_surface(backend, xv, yv, FT(0.5))
            smooth_surface_max_angle!(surf, 5.0)  # 5 degrees
            @test all(Array(surf.topo) .≈ FT(0.5))

        end

    end

    # ═════════════════════════════════════════════════════════
    # Lightweight mock for RockRatio (avoids JustRelax dependency)
    struct MockRockRatio3D{T <: AbstractArray}
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
            AT(zeros(FT, nx, ny, nz)),
            AT(zeros(FT, nx + 1, ny + 1, nz + 1)),
            AT(zeros(FT, nx + 1, ny, nz)),
            AT(zeros(FT, nx, ny + 1, nz)),
            AT(zeros(FT, nx, ny, nz + 1)),
            AT(zeros(FT, nx + 1, ny + 1, nz)),
            AT(zeros(FT, nx, ny + 1, nz + 1)),
            AT(zeros(FT, nx + 1, ny, nz + 1)),
        )
    end

    function control_bounds(x, i, dual)
        dual || return x[i], x[i + 1]
        lo = i == 1 ? x[1] : (x[i - 1] + x[i]) / 2
        hi = i == length(x) ? x[end] : (x[i] + x[i + 1]) / 2
        return lo, hi
    end

    function plane_fraction(ratio, xv, yv, zv, px, py, pz, a, bx, by)
        expected = similar(Array(ratio))
        for k in axes(expected, 3), j in axes(expected, 2), i in axes(expected, 1)
            xlo, xhi = control_bounds(xv, i, px)
            ylo, yhi = control_bounds(yv, j, py)
            zlo, zhi = control_bounds(zv, k, pz)
            height = a + bx * (xlo + xhi) / 2 + by * (ylo + yhi) / 2
            expected[i, j, k] = clamp((height - zlo) / (zhi - zlo), zero(FT), one(FT))
        end
        return expected
    end

    @testset "MarkerSurface — Rock fraction (compute_rock_fraction!)" begin
        xv, yv, zv = make_grid(; nz = 4)
        nx, ny, nz_g = length(xv) - 1, length(yv) - 1, length(zv) - 1
        di = (xv[2] - xv[1], yv[2] - yv[1], zv[2] - zv[1])

        @testset "Surface above domain → all rock" begin
            surf = init_marker_surface(backend, xv, yv, FT(1.5))  # above ztop=1.0
            ϕ = MockRockRatio3D(nx, ny, nz_g)
            compute_rock_fraction!(ϕ, surf, (xv, yv, zv), di)
            @test all(Array(ϕ.center) .≈ 1.0)
        end

        @testset "Surface below domain → all air" begin
            surf = init_marker_surface(backend, xv, yv, FT(-0.5))  # below zbot=0.0
            ϕ = MockRockRatio3D(nx, ny, nz_g)
            compute_rock_fraction!(ϕ, surf, (xv, yv, zv), di)
            @test all(Array(ϕ.center) .≈ 0.0)
        end

        @testset "Surface at mid-height → partial fill" begin
            surf = init_marker_surface(backend, xv, yv, FT(0.5))
            ϕ = MockRockRatio3D(nx, ny, nz_g)
            compute_rock_fraction!(ϕ, surf, (xv, yv, zv), di)
            center_cpu = Array(ϕ.center)
            for k in 1:nz_g
                if zv[k + 1] ≤ 0.5
                    # Cells entirely below surface → fully rock
                    @test all(center_cpu[:, :, k] .≈ 1.0)
                elseif zv[k] ≥ 0.5
                    # Cells entirely above surface → fully air (rock = 0)
                    @test all(center_cpu[:, :, k] .≈ 0.0)
                end
            end
        end

        @testset "Staggered nodes are consistent" begin
            surf = init_marker_surface(backend, xv, yv, FT(0.5))
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

        @testset "Every placement uses its own control volume" begin
            surf = init_marker_surface(backend, xv, yv, FT(0.4))
            ϕ = MockRockRatio3D(nx, ny, nz_g)
            compute_rock_fraction!(ϕ, surf, (xv, yv, zv), di)

            cell_profile = FT[1, 0.6, 0, 0]
            dual_profile = FT[1, 1, 0.1, 0, 0]
            for ratio in (ϕ.center, ϕ.Vx, ϕ.Vy, ϕ.xy)
                @test Array(ratio)[1, 1, :] ≈ cell_profile atol = eps(FT) * 16
            end
            for ratio in (ϕ.vertex, ϕ.Vz, ϕ.yz, ϕ.xz)
                @test Array(ratio)[1, 1, :] ≈ dual_profile atol = eps(FT) * 16
            end
            @test Array(ϕ.Vz)[1, 1, 3] ≈ FT(0.1) atol = eps(FT) * 16
        end

        @testset "Inclined surface respects x/y placement symmetry" begin
            xv, yv, zv = make_grid(; nx = 4, ny = 4, nz = 4)
            z_init = [FT(0.2) + FT(0.2) * (x + y) for x in xv, y in yv]
            surf = init_marker_surface(backend, xv, yv, z_init)
            ϕ = MockRockRatio3D(4, 4, 4)
            di = (xv[2] - xv[1], yv[2] - yv[1], zv[2] - zv[1])
            compute_rock_fraction!(ϕ, surf, (xv, yv, zv), di)

            @test Array(ϕ.Vx) ≈ permutedims(Array(ϕ.Vy), (2, 1, 3)) atol = eps(FT) * 32
            @test Array(ϕ.xz) ≈ permutedims(Array(ϕ.yz), (2, 1, 3)) atol = eps(FT) * 32
        end

        @testset "Inclined surface is exact at every placement under refinement" begin
            a, bx, by = FT(0.1), FT(0.1), FT(0.1)
            for n in (4, 8)
                xv = range(zero(FT), one(FT), n + 1)
                yv = range(zero(FT), one(FT), n + 1)
                zv = range(zero(FT), one(FT), 2)
                surf = init_marker_surface(backend, xv, yv, [a + bx * x + by * y for x in xv, y in yv])
                ϕ = MockRockRatio3D(n, n, 1)
                di = (xv[2] - xv[1], yv[2] - yv[1], one(FT))
                compute_rock_fraction!(ϕ, surf, (xv, yv, zv), di)

                for (ratio, px, py, pz) in (
                        (ϕ.center, false, false, false),
                        (ϕ.vertex, true, true, true),
                        (ϕ.Vx, true, false, false),
                        (ϕ.Vy, false, true, false),
                        (ϕ.Vz, false, false, true),
                        (ϕ.xy, true, true, false),
                        (ϕ.yz, false, true, true),
                        (ϕ.xz, true, false, true),
                    )
                    @test Array(ratio) ≈ plane_fraction(ratio, xv, yv, zv, px, py, pz, a, bx, by) atol = eps(FT) * 32
                end
            end
        end

        @testset "MarkerSurface Extended — Rock fraction consistency" begin
            xv, yv, zv = make_grid(; nx = 4, ny = 4, nz = 8)
            nx_g, ny_g, nz_g = 4, 4, length(zv) - 1
            di = (xv[2] - xv[1], yv[2] - yv[1], zv[2] - zv[1])

            # Flat surface at z=0.5 → cells below should be rock, above should be air
            surf = init_marker_surface(backend, xv, yv, FT(0.5))
            ϕ = MockRockRatio3D(nx_g, ny_g, nz_g)
            compute_rock_fraction!(ϕ, surf, (xv, yv, zv), di)

            # Check that the rock fraction sums approximately to 50% of total volume
            total_rock = sum(Array(ϕ.center)) / (nx_g * ny_g * nz_g)
            @test abs(total_rock - 0.5) < 0.1  # roughly half-filled

            # Monotonicity: lower cells should have higher rock fraction
            center_cpu = Array(ϕ.center)
            for i in 1:nx_g, j in 1:ny_g
                for k in 1:(nz_g - 1)
                    @test center_cpu[i, j, k] ≥ center_cpu[i, j, k + 1] - 1.0e-10
                end
            end
        end

    end


    @testset "MarkerSurface — Triangular prism intersection" begin
        # Verify the prism intersection gives correct rock fractions
        # for a simple geometry: triangle (0,0)-(1,0)-(0.5,0.5), area 0.25,
        # clipped against the cell's z-range [0, 1].

        @testset "Full prism below surface" begin
            val = JustPIC._triangle_rock_fraction(0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.5, 0.5, 2.0, 1.0, 0.0, 1.0)
            @test val ≈ 0.25 atol = 1.0e-10
        end

        @testset "Full prism above surface" begin
            val = JustPIC._triangle_rock_fraction(0.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.5, 0.5, -1.0, 1.0, 0.0, 1.0)
            @test val ≈ 0.0 atol = 1.0e-10
        end
    end
end

@testset "MarkerSurface — ImplicitGlobalGrid (single-rank periodic)" begin
    nx, ny, nz = 8, 8, 8
    init_global_grid(nx, ny, nz; periodx = 1, periody = 1, quiet = true)

    xv, yv, zv = make_grid(; nx = nx, ny = ny, nz = nz)
    nx1, ny1 = nx + 1, ny + 1
    AT = TA(backend)

    # vertex arrays share `ol` lines between periodic neighbours
    # (default overlap of 2 cells -> 3 vertex lines)
    ol = 3

    @testset "Halo exchange (periodic self-copy)" begin
        surf = init_marker_surface(backend, xv, yv, zero(FT))
        A = rand(FT, nx1, ny1)  # seams deliberately inconsistent
        set_topo_from_array!(surf, AT(A))

        update_surface_halo!(surf)
        T = Array(surf.topo)

        # periodic identification: line i ≡ line i + (nx1 - ol)
        @test T[1, :] ≈ T[nx1 - ol + 1, :]
        @test T[nx1, :] ≈ T[ol, :]
        @test T[:, 1] ≈ T[:, ny1 - ol + 1]
        @test T[:, ny1] ≈ T[:, ol]
    end

    @testset "Advection drivers run under an active global grid" begin
        surf = init_marker_surface(backend, xv, yv, FT(0.5))
        nxg, nyg, nzg = length(xv), length(yv), length(zv)
        grid_vxi = device_grid(staggered_velocity_grid(xv, yv, zv))
        V = sample_velocity(grid_vxi) do d, x, y, z
            d == 3 ? FT(0.1) : zero(FT)
        end

        dt = 0.1
        advect_marker_surface!(surf, V, grid_vxi, dt)
        @test all(abs.(Array(surf.topo) .- FT(0.51)) .≤ eps(FT) * 8)

    end

    finalize_global_grid(; finalize_MPI = false)
end
